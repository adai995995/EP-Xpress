// Direct inter-node GIN with producer-owned host staging inside each node.
// NCCL owns network progress; this file never enables or dereferences GPU P2P.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nccl.h>
#include <nccl_device.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
static void nc(ncclResult_t e) {
    if (e != ncclSuccess) throw std::runtime_error(ncclGetErrorString(e));
}
static void cu(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

struct Layout {
    int rank, world, local_rank, hidden, topk, tokens, experts, contexts;
    size_t stride, tx, rx, ctx, crx, host_combine;
    char* base;
    char** host;
    int* local; // global rank -> node-local index, -1 for a different node
    int* expert_rank;
    int* expert_local;
    int* prefix;
    __device__ int* send_counts() const { return reinterpret_cast<int*>(base); }
    __device__ int* recv_counts() const { return reinterpret_cast<int*>(base) + world; }
    __device__ char* tx_row(int t) const { return base + tx + size_t(t) * stride; }
    __device__ char* rx_row(int src, int t) const {
        return base + rx + (size_t(src) * tokens + t) * stride;
    }
    __device__ size_t result_offset(int src, int t) const {
        return crx + (size_t(src) * tokens + t) * hidden * 2;
    }
};

__device__ uint32_t load_word(const void* p) {
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ void require_device(bool condition) {
    // Memory safety/routing contract, not optional per-token diagnostics.
    if (!condition) asm volatile("trap;");
}

__global__ void phase_barrier(ncclDevComm dc, uint64_t timeout_cycles) {
    // Puts can originate on any context. A context-0-only signal is insufficient:
    // it can overtake payload writes on a different QP.
    ncclGinBarrierSession<ncclCoopCta> session(
        ncclCoopCta(), ncclGinAllContexts{dc}, ncclTeamTagWorld(), 0);
    auto result = session.sync(ncclCoopCta(), cuda::memory_order_acq_rel,
                              ncclGinFenceLevel::Put, timeout_cycles);
    require_device(result == ncclSuccess);
}

__global__ void pack_dispatch(Layout a, const uint32_t* hidden, const int64_t* ids,
                              const float* weights, uint8_t* expected, int n) {
    __shared__ int seen[256];
    __shared__ int host_needed;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *reinterpret_cast<volatile int*>(a.host[a.rank]) = n;
    for (int token = blockIdx.x; token < n; token += gridDim.x) {
        for (int p = threadIdx.x; p < a.world; p += blockDim.x) seen[p] = 0;
        if (threadIdx.x == 0) host_needed = 0;
        __syncthreads();
        for (int k = threadIdx.x; k < a.topk; k += blockDim.x) {
            int64_t e = ids[size_t(token) * a.topk + k];
            float w = weights[size_t(token) * a.topk + k];
            require_device(e >= -1 && e < a.experts && isfinite(w));
            if (e >= 0) atomicExch(seen + a.expert_rank[e], 1);
        }
        __syncthreads();
        for (int p = threadIdx.x; p < a.world; p += blockDim.x) {
            expected[size_t(token) * a.world + p] = seen[p];
            if (seen[p] && p != a.rank && a.local[p] >= 0) atomicExch(&host_needed, 1);
        }
        __syncthreads();
        char* gpu = a.tx_row(token);
        char* host = a.host[a.rank] + 128 + size_t(token) * a.stride;
        for (int j = threadIdx.x; j < a.hidden / 2; j += blockDim.x) {
            uint32_t v = hidden[size_t(token) * (a.hidden / 2) + j];
            reinterpret_cast<uint32_t*>(gpu)[j] = v;
            if (host_needed) reinterpret_cast<uint32_t*>(host)[j] = v;
        }
        int* gid = reinterpret_cast<int*>(gpu + a.hidden * 2);
        int* hid = reinterpret_cast<int*>(host + a.hidden * 2);
        float* gw = reinterpret_cast<float*>(gid + a.topk);
        float* hw = reinterpret_cast<float*>(hid + a.topk);
        for (int k = threadIdx.x; k < a.topk; k += blockDim.x) {
            gid[k] = int(ids[size_t(token) * a.topk + k]);
            gw[k] = weights[size_t(token) * a.topk + k];
            if (host_needed) { hid[k] = gid[k]; hw[k] = gw[k]; }
        }
        if (threadIdx.x == 0) {
            reinterpret_cast<int*>(gw + a.topk)[0] = token;
            reinterpret_cast<int*>(hw + a.topk)[0] = host_needed ? token : -1;
        }
        __syncthreads();
    }
    __threadfence_system(); // every writer, not just lane 0
}

__global__ void send_dispatch(Layout a, ncclDevComm dc, ncclWindow_t win,
                              const uint8_t* expected, int n) {
    ncclGin gin(dc, blockIdx.x % a.contexts);
    for (int token = blockIdx.x; token < n; token += gridDim.x) {
        for (int peer = threadIdx.x; peer < a.world; peer += blockDim.x) {
            if (a.local[peer] >= 0 || !expected[size_t(token) * a.world + peer]) continue;
            int slot = atomicAdd(a.send_counts() + peer, 1);
            require_device(slot < a.tokens);
            size_t dst = a.rx + (size_t(a.rank) * a.tokens + slot) * a.stride;
            gin.put(ncclTeamWorld(dc), peer, win, dst, win,
                    a.tx + size_t(token) * a.stride, a.stride, ncclGin_None{});
        }
    }
}

__global__ void send_counts(Layout a, ncclDevComm dc, ncclWindow_t win) {
    ncclGin gin(dc, 0);
    for (int peer = threadIdx.x; peer < a.world; peer += blockDim.x)
        if (a.local[peer] < 0)
            gin.put(ncclTeamWorld(dc), peer, win, (a.world + a.rank) * sizeof(int),
                    win, peer * sizeof(int), sizeof(int), ncclGin_None{});
}

__global__ void receive_prefix(Layout a, int n) {
    if (threadIdx.x) return;
    int total = 0;
    a.prefix[0] = 0;
    for (int src = 0; src < a.world; ++src) {
        int count = src == a.rank ? n : a.local[src] >= 0
            ? int(load_word(a.host[src])) : a.recv_counts()[src];
        require_device(count >= 0 && count <= a.tokens);
        total += count;
        a.prefix[src + 1] = total;
    }
}

__global__ void receive_dispatch(Layout a, uint32_t* output, int64_t* local_ids,
                                 float* weights, int64_t* inverse, int* count, int capacity) {
    __shared__ int row, token, hit;
    const int total = a.prefix[a.world];
    for (int task = blockIdx.x; task < total; task += gridDim.x) {
        int src = 0;
        while (a.prefix[src + 1] <= task) ++src;
        int index = task - a.prefix[src];
        const char* record = src == a.rank ? a.tx_row(index) : a.local[src] >= 0
            ? a.host[src] + 128 + size_t(index) * a.stride : a.rx_row(src, index);
        const int* ids = reinterpret_cast<const int*>(record + a.hidden * 2);
        const float* w = reinterpret_cast<const float*>(ids + a.topk);
        if (threadIdx.x == 0) {
            token = int(load_word(w + a.topk));
            hit = 0;
        }
        __syncthreads();
        int saved_token = token;
        __syncthreads(); // shared token may be replaced by the next loop iteration
        if (saved_token < 0) continue; // producer had no other local readers
        require_device(saved_token < a.tokens);
        for (int k = threadIdx.x; k < a.topk; k += blockDim.x) {
            int e = int(load_word(ids + k));
            if (e >= 0 && a.expert_rank[e] == a.rank) atomicExch(&hit, 1);
        }
        __syncthreads();
        bool selected = hit;
        __syncthreads();
        if (!selected) continue;
        if (threadIdx.x == 0) {
            row = atomicAdd(count, 1);
            require_device(row < capacity);
            inverse[size_t(row) * 2] = src;
            inverse[size_t(row) * 2 + 1] = saved_token;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < a.hidden / 2; j += blockDim.x)
            output[size_t(row) * (a.hidden / 2) + j] =
                load_word(reinterpret_cast<const uint32_t*>(record) + j);
        for (int k = threadIdx.x; k < a.topk; k += blockDim.x) {
            int e = int(load_word(ids + k));
            bool mine = e >= 0 && a.expert_rank[e] == a.rank;
            local_ids[size_t(row) * a.topk + k] = mine ? a.expert_local[e] : -1;
            weights[size_t(row) * a.topk + k] = mine ? __uint_as_float(load_word(w + k)) : 0.f;
        }
        __syncthreads();
    }
}

__global__ void pack_combine(Layout a, const uint32_t* partial, const int64_t* inverse,
                             const int* count, int capacity) {
    const int rows = *count;
    require_device(rows >= 0 && rows <= capacity);
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        int dst = int(inverse[size_t(row) * 2]);
        int token = int(inverse[size_t(row) * 2 + 1]);
        require_device(dst >= 0 && dst < a.world && token >= 0 && token < a.tokens);
        char* target = dst == a.rank ? a.base + a.result_offset(a.rank, token)
            : a.local[dst] >= 0 ? a.host[a.rank] + a.host_combine +
                (size_t(a.local[dst]) * a.tokens + token) * a.hidden * 2
            : a.base + a.ctx + size_t(row) * a.hidden * 2;
        for (int j = threadIdx.x; j < a.hidden / 2; j += blockDim.x)
            reinterpret_cast<uint32_t*>(target)[j] = partial[size_t(row) * (a.hidden / 2) + j];
    }
    __threadfence_system();
}

__global__ void send_combine(Layout a, ncclDevComm dc, ncclWindow_t win,
                             const int64_t* inverse, const int* count) {
    ncclGin gin(dc, blockIdx.x % a.contexts);
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < *count;
         row += blockDim.x * gridDim.x) {
        int dst = int(inverse[size_t(row) * 2]);
        if (a.local[dst] >= 0) continue;
        int token = int(inverse[size_t(row) * 2 + 1]);
        gin.put(ncclTeamWorld(dc), dst, win, a.result_offset(a.rank, token),
                win, a.ctx + size_t(row) * a.hidden * 2, a.hidden * 2, ncclGin_None{});
    }
}

__global__ void reduce_combine(Layout a, const uint8_t* expected, __nv_bfloat16* output, int n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(n) * (a.hidden / 2); i += size_t(blockDim.x) * gridDim.x) {
        int token = i / (a.hidden / 2), col = i % (a.hidden / 2);
        float x = 0.f, y = 0.f;
        // Global rank order, one FP32 accumulation, one final BF16 rounding.
        // Contributions already contain the expert routing weights.
        for (int peer = 0; peer < a.world; ++peer) {
            if (!expected[size_t(token) * a.world + peer]) continue;
            const char* row = peer != a.rank && a.local[peer] >= 0
                ? a.host[peer] + a.host_combine +
                    (size_t(a.local_rank) * a.tokens + token) * a.hidden * 2
                : a.base + a.result_offset(peer, token);
            uint32_t bits = load_word(reinterpret_cast<const uint32_t*>(row) + col);
            x += __bfloat162float(__ushort_as_bfloat16(bits & 0xffff));
            y += __bfloat162float(__ushort_as_bfloat16(bits >> 16));
        }
        output[i * 2] = __float2bfloat16(x);
        output[i * 2 + 1] = __float2bfloat16(y);
    }
}

class GinRuntime {
    ncclComm_t comm;
    ncclDevComm dc{};
    ncclWindow_t window = nullptr;
    Layout a{};
    std::vector<void*> registered;
    bool device_comm_created = false, closed = false;
    uint64_t timeout_cycles;
    int blocks;
    size_t bytes, host_bytes;
public:
    GinRuntime(uint64_t pointer, uint64_t owner_symbol, int hidden, int topk, int tokens,
               int contexts, double timeout, size_t gpu_budget, size_t host_budget,
               std::vector<int> local, std::vector<uint64_t> host,
               std::vector<int> expert_rank, std::vector<int> expert_local)
        : comm(reinterpret_cast<ncclComm_t>(pointer)) {
        if (owner_symbol != reinterpret_cast<uint64_t>(&ncclAllGather))
            throw std::runtime_error("GIN extension and borrowed communicator use different NCCL DSOs");
        int version; nc(ncclGetVersion(&version));
        if (version != NCCL_VERSION_CODE || version < 23007)
            throw std::runtime_error("GIN requires matching NCCL device headers/runtime >= 2.30.7");
        nc(ncclCommUserRank(comm, &a.rank)); nc(ncclCommCount(comm, &a.world));
        if (a.world > 256 || int(local.size()) != a.world || int(host.size()) != a.world ||
            expert_rank.empty() || expert_rank.size() != expert_local.size() ||
            hidden <= 0 || hidden % 2 || topk < 1 || topk > 256 || tokens < 1 ||
            contexts < 1 || contexts > 32)
            throw std::runtime_error("invalid GIN layout");
        a.local_rank = local[a.rank];
        int local_size = *std::max_element(local.begin(), local.end()) + 1;
        a.hidden = hidden; a.topk = topk; a.tokens = tokens; a.experts = expert_rank.size();
        a.stride = align_up(size_t(hidden) * 2 + size_t(topk) * 8 + 4, 128);
        a.tx = align_up(size_t(a.world) * 8, 128);
        a.rx = a.tx + size_t(tokens) * a.stride;
        a.ctx = a.rx + size_t(a.world) * tokens * a.stride;
        a.crx = a.ctx + size_t(a.world) * tokens * hidden * 2;
        bytes = align_up(a.crx + size_t(a.world) * tokens * hidden * 2, 2 << 20);
        a.host_combine = align_up(128 + size_t(tokens) * a.stride, 4096);
        host_bytes = align_up(a.host_combine + size_t(local_size) * tokens * hidden * 2, 4096);
        if (bytes > gpu_budget || host_bytes * local_size > host_budget)
            throw std::runtime_error("GIN staging exceeds configured GPU or node host budget");
        ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
        nc(ncclCommQueryProperties(comm, &props));
        if (!props.deviceApiSupport)
            throw std::runtime_error("NCCL communicator lacks device API/symmetric memory; "
                                     "set NCCL_CUMEM_ENABLE=1 before creating the parent communicator");
        ncclDevCommRequirements req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 32, 0)
        if (!props.ginSupport[NCCL_GIN_TYPE_GDAKI])
            throw std::runtime_error("GDAKI unavailable; no silent host-network fallback");
        req.ginType = NCCL_GIN_TYPE_GDAKI;
#else
        if (props.ginType != NCCL_GIN_TYPE_GDAKI)
            throw std::runtime_error("direct GDAKI unavailable; no silent host-network fallback");
#endif
        req.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        req.ginContextCount = contexts;
        req.worldGinBarrierCount = 1;
        try {
            nc(ncclDevCommCreate(comm, &req, &dc)); device_comm_created = true;
            a.contexts = dc.ginContextCount;
            nc(ncclMemAlloc(reinterpret_cast<void**>(&a.base), bytes));
            nc(ncclCommWindowRegister(comm, a.base, bytes, &window, NCCL_WIN_STRICT_ORDERING));
            std::vector<char*> mapped(a.world, nullptr);
            for (int p = 0; p < a.world; ++p) if (local[p] >= 0) {
                void* ptr = reinterpret_cast<void*>(host[p]);
                cu(cudaHostRegister(ptr, host_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
                registered.push_back(ptr);
                cu(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped[p]), ptr, 0));
            }
            cu(cudaMalloc(&a.host, a.world * sizeof(char*)));
            cu(cudaMemcpy(a.host, mapped.data(), a.world * sizeof(char*), cudaMemcpyHostToDevice));
            cu(cudaMalloc(&a.local, a.world * sizeof(int)));
            cu(cudaMemcpy(a.local, local.data(), a.world * sizeof(int), cudaMemcpyHostToDevice));
            cu(cudaMalloc(&a.expert_rank, a.experts * sizeof(int)));
            cu(cudaMemcpy(a.expert_rank, expert_rank.data(), a.experts * sizeof(int), cudaMemcpyHostToDevice));
            cu(cudaMalloc(&a.expert_local, a.experts * sizeof(int)));
            cu(cudaMemcpy(a.expert_local, expert_local.data(), a.experts * sizeof(int), cudaMemcpyHostToDevice));
            cu(cudaMalloc(&a.prefix, (a.world + 1) * sizeof(int)));
            int device; cu(cudaGetDevice(&device));
            cudaDeviceProp prop; cu(cudaGetDeviceProperties(&prop, device));
            blocks = std::min(64, prop.multiProcessorCount);
            int clock_khz; cu(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
            timeout_cycles = uint64_t(timeout * clock_khz * 1000.0);
        } catch (...) { close(); throw; }
    }
    void dispatch(uint64_t hidden, uint64_t ids, uint64_t weights, int n, int capacity,
                  uint64_t out, uint64_t out_ids, uint64_t out_weights,
                  uint64_t inverse, uint64_t count, uint64_t expected, uint64_t stream_ptr) {
        if (closed || n < 0 || n > a.tokens || capacity < 0 || size_t(capacity) > size_t(a.world) * a.tokens)
            throw std::runtime_error("invalid GIN dispatch capacity or closed runtime");
        auto s = reinterpret_cast<cudaStream_t>(stream_ptr);
        phase_barrier<<<1,256,0,s>>>(dc, timeout_cycles); // prior readers finished; safe to recycle
        // Every remote producer overwrites its count, including zero. Clearing
        // receive slots here could erase a faster peer's new publication.
        cu(cudaMemsetAsync(a.base, 0, a.world * sizeof(int), s));
        cu(cudaMemsetAsync(reinterpret_cast<void*>(count), 0, sizeof(int), s));
        if (capacity) {
            cu(cudaMemsetAsync(reinterpret_cast<void*>(out), 0, size_t(capacity) * a.hidden * 2, s));
            cu(cudaMemsetAsync(reinterpret_cast<void*>(out_ids), 0xff, size_t(capacity) * a.topk * 8, s));
            cu(cudaMemsetAsync(reinterpret_cast<void*>(out_weights), 0, size_t(capacity) * a.topk * 4, s));
        }
        int grid = std::max(1, std::min(blocks, n));
        pack_dispatch<<<grid,128,0,s>>>(a, reinterpret_cast<uint32_t*>(hidden),
            reinterpret_cast<int64_t*>(ids), reinterpret_cast<float*>(weights),
            reinterpret_cast<uint8_t*>(expected), n);
        send_dispatch<<<grid,32,0,s>>>(a, dc, window, reinterpret_cast<uint8_t*>(expected), n);
        send_counts<<<1,256,0,s>>>(a, dc, window);
        phase_barrier<<<1,256,0,s>>>(dc, timeout_cycles); // remote payload/count and host publication
        receive_prefix<<<1,32,0,s>>>(a, n);
        receive_dispatch<<<blocks,128,0,s>>>(a, reinterpret_cast<uint32_t*>(out),
            reinterpret_cast<int64_t*>(out_ids), reinterpret_cast<float*>(out_weights),
            reinterpret_cast<int64_t*>(inverse), reinterpret_cast<int*>(count), capacity);
        cu(cudaGetLastError());
    }
    void combine(uint64_t partial, uint64_t inverse, uint64_t count, uint64_t expected,
                 uint64_t output, int n, int capacity, uint64_t stream_ptr) {
        if (closed || n < 0 || n > a.tokens || capacity < 0 || size_t(capacity) > size_t(a.world) * a.tokens)
            throw std::runtime_error("invalid GIN combine capacity or closed runtime");
        auto s = reinterpret_cast<cudaStream_t>(stream_ptr);
        phase_barrier<<<1,256,0,s>>>(dc, timeout_cycles); // prior result readers finished
        pack_combine<<<blocks,128,0,s>>>(a, reinterpret_cast<uint32_t*>(partial),
            reinterpret_cast<int64_t*>(inverse), reinterpret_cast<int*>(count), capacity);
        send_combine<<<a.contexts,32,0,s>>>(a, dc, window,
            reinterpret_cast<int64_t*>(inverse), reinterpret_cast<int*>(count));
        phase_barrier<<<1,256,0,s>>>(dc, timeout_cycles);
        reduce_combine<<<blocks,256,0,s>>>(a, reinterpret_cast<uint8_t*>(expected),
            reinterpret_cast<__nv_bfloat16*>(output), n);
        cu(cudaGetLastError());
    }
    py::dict info() const {
        py::dict d; d["gpu_bytes"] = bytes; d["host_bytes_per_producer"] = host_bytes;
        d["gin_contexts"] = a.contexts; d["nccl_version"] = NCCL_VERSION_CODE;
        d["local_payload"] = "mapped_host"; d["remote_payload"] = "GDAKI"; return d;
    }
    void close() {
        if (closed) return;
        // Collective destruction is explicit. Never initiate it from a GC destructor.
        cu(cudaDeviceSynchronize());
        if (window) { nc(ncclCommWindowDeregister(comm, window)); window = nullptr; }
        if (a.base) { nc(ncclMemFree(a.base)); a.base = nullptr; }
        for (void* p : registered) cu(cudaHostUnregister(p));
        registered.clear();
        for (void* p : {static_cast<void*>(a.host), static_cast<void*>(a.local),
                       static_cast<void*>(a.expert_rank), static_cast<void*>(a.expert_local),
                       static_cast<void*>(a.prefix)}) if (p) cu(cudaFree(p));
        if (device_comm_created) { nc(ncclDevCommDestroy(comm, &dc)); device_comm_created = false; }
        closed = true;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<GinRuntime>(m, "GinRuntime")
        .def(py::init<uint64_t,uint64_t,int,int,int,int,double,size_t,size_t,
                      std::vector<int>,std::vector<uint64_t>,std::vector<int>,std::vector<int>>())
        .def("dispatch", &GinRuntime::dispatch).def("combine", &GinRuntime::combine)
        .def("info", &GinRuntime::info).def("close", &GinRuntime::close);
}

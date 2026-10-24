# Opt-in full decode Graph; use identical graph buckets in the NCCL baseline.
# Requires --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 8.
# Prefill remains eager. Source from bash before starting the patched server.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_same_stream.sh"
export EP_XPRESS_CUDA_GRAPH=1
export EP_XPRESS_DECODE_PEER_BATCH=2
export EP_XPRESS_DECODE_CHANNELS=0
export EP_XPRESS_PREFILL_CHANNELS=0

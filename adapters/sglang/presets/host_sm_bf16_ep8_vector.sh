# Opt-in measured candidate; end-to-end gains are modest and workload-specific.
# Source this file from bash before launching the patched SGLang server.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8.sh"
export EP_XPRESS_PREFILL_READ_UNROLL=4
export EP_XPRESS_PREFILL_WRITE_UNROLL=4
export EP_XPRESS_DISPATCH_READ_UNROLL=4

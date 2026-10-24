# Measured eager decode candidate; preserves the vector/LAST prefill settings.
# Source from bash before launching the patched SGLang server.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_last_data.sh"
export EP_XPRESS_DECODE_SAME_STREAM=1

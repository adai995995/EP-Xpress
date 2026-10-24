# Opt-in measured candidate: vector copies plus payload-bearing LAST frames.
# Source from bash; preserve all other model and host-only baseline settings.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_vector.sh"
export EP_XPRESS_COMBINE_LAST_DATA=1

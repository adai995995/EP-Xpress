# Opt-in dispatch READY-first scheduling for bounded BF16 EP8 routing.
# See docs/model-performance.md for measured scope and variability.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_graph_read_progress.sh"
export EP_XPRESS_READY_CACHE=1
export EP_XPRESS_PACKET_READY_CACHE=0
export EP_XPRESS_RX_POLL_WARP=0

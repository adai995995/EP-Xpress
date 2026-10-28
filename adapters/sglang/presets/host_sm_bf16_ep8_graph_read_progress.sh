# Opt-in receive-fence scope refinement, measured with dynamic BF16 EP8 routing.
# Preserve the direct-slot preset's channels, packet sizes and Graph settings.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_graph_direct.sh"
export EP_XPRESS_RX_FENCE=1
# Independent model ablations did not establish gains from these options.
export EP_XPRESS_ROUTE_PACKED=0
export EP_XPRESS_READY_BATCH=0
export EP_XPRESS_PACKET_RECV_LOCAL=0

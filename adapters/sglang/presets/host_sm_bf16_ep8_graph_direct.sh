# Opt-in direct-slot dispatch and independent packet progress.
# Same BF16 EP8 model/Graph settings as the small-decode preset. Dynamic routes
# are rebuilt on every invocation; this does not select a universal channel policy.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_graph_small_decode.sh"
export EP_XPRESS_DISPATCH_DIRECT=1
# Extra metadata/ACK warps did not improve measured B1/B32 model throughput.
export EP_XPRESS_DISPATCH_RETIRE=0
export EP_XPRESS_PACKET_PIPELINE=1
# Large generic-ring dispatch improved in the standalone shape sweep.
# Model prefill TTFT benefit is not established; see docs/model-performance.md.
export EP_XPRESS_RING_SPLIT=1

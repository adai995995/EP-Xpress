# Measured small-decode candidate: eight H20D GPUs, H=2048, global B1/B32.
# Larger bounded-decode batches can regress; this is not an automatic policy.
# Source from bash before worker startup. Keep eager prefill and Graph buckets
# 1/2/4/8, and retain the same model/framework settings as the archived baseline.
source "$(dirname -- "${BASH_SOURCE[0]}")/host_sm_bf16_ep8_graph.sh"
export EP_XPRESS_HOST_BUDGET_BYTES=1073741824
export EP_XPRESS_DECODE_CHANNELS=2
export EP_XPRESS_DECODE_DISPATCH_INDEXED=1
export EP_XPRESS_DECODE_DISPATCH_PIPELINE=1
export EP_XPRESS_DECODE_DESCRIPTOR_BATCH=32
export EP_XPRESS_DECODE_PUBLISH_BATCH=8
export EP_XPRESS_DISPATCH_SPLIT=1
export EP_XPRESS_DISPATCH_DEFER_RECLAIM=0
export EP_XPRESS_REUSE_RETURN_INDEX=1
export EP_XPRESS_HOST_READ64=1
export EP_XPRESS_DISPATCH_WRITE_UNROLL=4
export EP_XPRESS_DECODE_PACKET_ROWS=4
export EP_XPRESS_DECODE_PACKET_CHANNELS=8
export EP_XPRESS_DECODE_CONTROL_RECORDS=0
export EP_XPRESS_DECODE_ACK_BATCH=1
export EP_XPRESS_DECODE_CONTROL_WARP=0
export EP_XPRESS_DECODE_READY_PREFETCH=0
export EP_XPRESS_DECODE_BALANCED_SEND=0
export EP_XPRESS_DECODE_SPLIT_TX_RX=0
export EP_XPRESS_DECODE_DEFER_RECLAIM=0

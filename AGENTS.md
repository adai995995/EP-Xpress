# Contributor instructions

- Keep this repository independent of inference frameworks. Framework-specific
  imports and patches belong in adapters; CUDA transport belongs in the core.
- Preserve inference-only BF16 dispatch/combine contracts and apply routing
  weights exactly once. Do not silently enable GPU peer access, NVLink payloads,
  RDMA, CPU reduction, or CUDA Graph modes that are not implemented.
- Develop source locally and validate CUDA changes on an operator-provided
  development machine. Keep its address, access configuration, deployment paths,
  model paths, and raw logs outside this repository. Do not embed credentials.
- Use generic public Git author and committer metadata:
  `EP-Xpress Contributors <contributors@users.noreply.github.com>`.
- Enable `.githooks` and run `python tools/check_public_content.py --worktree
  --staged --history` before every push. Review the staged diff as well. Never
  bypass the checks to publish sensitive information, including deleted content
  still present in history. Keep organization-specific deny patterns outside
  the repository and supply them through `EP_XPRESS_PRIVATE_PATTERNS_FILE`.
- Test changed contracts and failure paths. For CUDA transport changes, run
  native tests and distributed replay with multiple chunks, empty ranks, and
  slot reuse. Synthetic replay timing is not model inference performance.
- Keep comparisons on the same GPU configuration, dtype, routing trace, expert
  kernel, and Graph mode. Clearly distinguish tested functionality from planned
  optimization and unverified framework integration.

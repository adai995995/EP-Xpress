# Contributor instructions

- Develop directly on `main`. Create a separate Git branch only when the user
  explicitly requests one.
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
- Prioritize full-model SGLang end-to-end performance measurements. For tuning
  within an already validated execution path, proceed directly to the model
  comparison rather than rerunning broad native or distributed functional suites.
  Use only focused checks when a change creates a concrete correctness or hang
  risk, especially routing, buffer lifetime, ACK/progress protocols, or interfaces.
  Expand testing only when failures or unresolved concerns justify it. Synthetic
  replay timing is not model inference performance.
- The performance baseline is SGLang's default NCCL backend. Unless explicitly
  requested otherwise, disable NVLink and GPU P2P for both candidates and baseline;
  keep the GPU configuration and workload identical.
- Reuse the archived multi-run NCCL baseline for subsequent EP-only changes.
  Record arithmetic means and observed variation. Rebuild that baseline only
  when hardware, software, model, workload, or execution settings change, or
  evidence shows the saved baseline is no longer comparable.
- Keep comparisons on the same GPU configuration, dtype, routing trace, expert
  kernel, and Graph mode. Clearly distinguish tested functionality from planned
  optimization and unverified framework integration.

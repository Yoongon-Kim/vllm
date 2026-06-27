# H200 throughput bench — vLLM side

This branch (`h200-bench`) = `main` + the decode-throughput **bench wrapper scripts**,
pinning the exact vLLM commit used for the paper's efficiency numbers.

**Full runbook** (env, model download, calibration, bench): see `H200_BENCH_SETUP.md`
in the **LRoSA-dev** repo. vLLM-specific steps:

1. **Build for sm90 (Hopper)** — the CUDA ext (incl. the LRoSA radix top-k in
   `csrc/sampler.cu`) must be compiled on the H200:
   ```bash
   export TORCH_CUDA_ARCH_LIST="9.0"
   pip install -e . --no-build-isolation
   ```
2. **Bench**: `decode_latency_bench.py` (committed) drives one backend/ctx/batch per
   invocation; `run_tput_qwen8_fkv_lrosa.sh` sweeps the FKV/LRoSA × ctx × batch matrix
   on 3 GPUs (override `PY` / `HF_HOME` / `TMPDIR` / `PCA_REPO` via env). Headline metric
   = `AGG_TOK_S` at FIXED batch.

Note: the source box's uncommitted working-tree changes (TriAttention faithful port,
opt-in `LROSA_RADIX_SPLIT`/`FUSED_FIXUP` perf experiments — all default-OFF) are
intentionally NOT on this branch; the throughput bench uses only committed paths.

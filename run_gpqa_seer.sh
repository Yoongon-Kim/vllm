#!/usr/bin/env bash
# run_gpqa_seer.sh <token_budget> <gpu>
# SeerAttention-R GPQA (Qwen3-8B, native rope) on one GPU. 8 attempts/problem ->
# pass@1, batched (batch_size 16, ~7-10x vs one-at-a-time), resume on.
# token_budget is the seer analog of n_fac. Uses the `seer` conda env (NOT vllm).
# Results: <PCA>/results/gpqa/Qwen3-8B/seer_attention_nf<budget>/ ; engine="seer".
tb=${1:?token_budget}; g=${2:?gpu}
cd /NHNHOME/jiwonsong/LRoSA-dev
export TMPDIR=/NHNHOME/jiwonsong/tmp HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub HF_HUB_OFFLINE=1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
SEER_PY=$HOME/miniforge3/envs/seer/bin/python
CUDA_VISIBLE_DEVICES=$g "$SEER_PY" -m eval.gpqa_seer \
  --token_budget "$tb" --num_runs 8 --num_samples 0 --max_new_tokens 38912 \
  --batch_size 16 --cuda_device 0 \
  2>&1 | tee "/NHNHOME/jiwonsong/tmp/gpqa_seer_${tb}.log"

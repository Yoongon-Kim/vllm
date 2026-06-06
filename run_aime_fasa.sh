#!/usr/bin/env bash
# AIME25 / Qwen3-8B (thinking, NATIVE rope) / FASA-fc (n_tip=16, n_fac=2048)
# on GPU 0. 8 attempts/problem -> pass@1. Resumable. Run AFTER SqueezedAttention
# frees GPU 0. Uses the native fasa_idom basis (re-calibrated --no_yarn).
# Results: <PCA_REPO>/results/aime25/Qwen3-8B/fasa_nt16/
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
set -u
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub   # Qwen3-8B here; datasets in default cache
PY=$HOME/miniforge3/envs/vllm/bin/python

CUDA_VISIBLE_DEVICES=0 VLLM_CACHE_ROOT="$HOME/.cache/aime_fasa" \
  "$PY" reasoning_vllm_eval.py --eval aime25 --mode fasa --model Qwen/Qwen3-8B \
    --n_tip 16 --n_fac 2048 \
    --num_runs 8 --num_samples 0 --max_new_tokens 38912 \
    --max_num_seqs 8 --gpu_mem 0.85 \
  2>&1 | tee /NHNHOME/jiwonsong/tmp/aime_fasa.log

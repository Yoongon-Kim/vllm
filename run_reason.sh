#!/usr/bin/env bash
# run_reason.sh <eval> <mode> <n_fac> <gpu>
# One Qwen3-8B (thinking, NATIVE rope) reasoning run on one GPU. 8 attempts ->
# pass@1, resume on. eval in {aime25,math500,gpqa}; mode in {fkv,lrosa,loki,
# quest,fasa}. n_fac ignored for fkv. Results: <PCA>/results/<eval>/Qwen3-8B/<run_name>/
ev=$1; md=$2; nf=$3; g=$4
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
set -u
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/NHNHOME/jiwonsong/hf_cache/hub
PY=$HOME/miniforge3/envs/vllm/bin/python

case "$md" in
  fkv)   extra="";                       rn="fkv";;
  lrosa) extra="--cs_h 32 --n_fac $nf";  rn="lrosa_cs32_fp8_nf$nf";;
  loki)  extra="--cs_h 32 --n_fac $nf";  rn="loki_cs32_fp8_nf$nf";;
  quest) extra="--n_fac $nf";            rn="quest_nf$nf";;
  fasa)  extra="--n_tip 16 --n_fac $nf"; rn="fasa_nt16_nf$nf";;
  *) echo "bad mode $md"; exit 1;;
esac

CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT="$HOME/.cache/r_${ev}_${md}_${nf}" \
  "$PY" reasoning_vllm_eval.py --eval "$ev" --mode "$md" $extra --model Qwen/Qwen3-8B \
    --num_runs 8 --num_samples 0 --max_new_tokens 38912 --max_num_seqs 8 --gpu_mem 0.85 \
    --run_name "$rn" \
  2>&1 | tee "/NHNHOME/jiwonsong/tmp/r_${ev}_${md}_${nf}.log"

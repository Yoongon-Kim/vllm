#!/usr/bin/env bash
# Throughput matrix: Qwen3-8B, FKV vs LRoSA, ctx {4k..128k} x batch {1,2,4,8},
# budget(n_fac)=2048, YaRN. Headline metric = AGG_TOK_S (throughput) at FIXED
# batch (clean: speedup ratio == latency ratio). LRoSA = deployed (fp8 cs32).
# 3 GPUs (0/1/2), distinct VLLM_PORT per concurrent slot, fresh per-slot cache.
set -u
# Portable: override PY / HF_HOME / TMPDIR / PCA_REPO via env on a new box (e.g. H200).
PY=${PY:-python}                                   # activate the vLLM conda env, or set PY=/path/to/python
cd "$(dirname "$(readlink -f "$0")")"              # vLLM repo root (this script lives there)
export TMPDIR=${TMPDIR:-/tmp} LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export HF_HOME=${HF_HOME:-/NHNHOME/jiwonsong/hf_cache} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=${OUT:-$TMPDIR/tput_qwen8}; mkdir -p $OUT
SUM=$OUT/summary.log
PCA_REPO=${PCA_REPO:-/NHNHOME/jiwonsong/LRoSA-dev}
MODEL=${MODEL:-Qwen/Qwen3-8B}
NFAC=${NFAC:-2048}; DECODE=${DECODE:-64}; CSH=${CSH:-32}
BASIS=${BASIS:-$PCA_REPO/bases/qwen3_8b/pca_d1_cs32_kv_head_qwen3_8b.pt}
declare -A CTX=( [4k]=4096 [8k]=8192 [16k]=16384 [32k]=32768 [64k]=65536 [128k]=130816 )
GPUS=(0 1 2)

run_one(){ local be=$1 label=$2 b=$3 g=$4 port=$5 plen=${CTX[$2]} f=$OUT/${1}_${2}_b${3}.txt
  local extra=""
  case $be in
    lrosa) extra="--basis $BASIS --cs_h $CSH --fp8_projk";;
    fasa)  extra="--n_tip 16";;
  esac
  local CR=$HOME/.cache/vllm_tp_${be}_${g}; rm -rf "$CR"
  CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$port VLLM_CACHE_ROOT=$CR \
    TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_tp_${be}_${g} TRITON_CACHE_DIR=$TMPDIR/tri_tp_${be}_${g} \
    $PY decode_latency_bench.py --backend $be --model $MODEL --prefill_len $plen \
      --decode_len $DECODE --n_fac $NFAC --batch_size $b --gpu_mem 0.90 $extra \
      > "$f" 2>&1
}

# job order: backend-major so FKV then LRoSA finish first; then ctx, then batch.
jobs=()
for be in fkv lrosa; do for label in 4k 8k 16k 32k 64k 128k; do for b in 1 2 4 8; do
  jobs+=("$be:$label:$b"); done; done; done

echo "=== qwen8 throughput matrix start $(date +%H:%M:%S)  (${#jobs[@]} cells) ===" | tee -a $SUM
i=0; ng=${#GPUS[@]}
while [ $i -lt ${#jobs[@]} ]; do
  pids=(); slots=()
  for ((s=0; s<ng && i<${#jobs[@]}; s++)); do
    IFS=: read -r be label b <<< "${jobs[$i]}"
    g=${GPUS[$s]}; port=$((45000+s*100))
    run_one "$be" "$label" "$b" "$g" "$port" & pids+=($!); slots+=("${jobs[$i]}")
    i=$((i+1))
  done
  wait "${pids[@]}"
  for j in "${slots[@]}"; do
    IFS=: read -r be label b <<< "$j"; f=$OUT/${be}_${label}_b${b}.txt
    agg=$(grep -hoE "AGG_TOK_S=[0-9.]+" $f 2>/dev/null | tail -1 | cut -d= -f2)
    ps=$(grep -hoE "PER_STREAM_TOK_S=[0-9.]+" $f 2>/dev/null | tail -1 | cut -d= -f2)
    ms=$(grep -hoE "DECODE_MS_PER_TOK=[0-9.]+" $f 2>/dev/null | tail -1 | cut -d= -f2)
    conc=$(grep -hoE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" $f 2>/dev/null | tail -1 | grep -oE "[0-9.]+x$")
    if [ -n "$agg" ]; then
      flag=""; cn=${conc%x}
      awk "BEGIN{exit !($cn < $b)}" 2>/dev/null && flag=" !!PREEMPT(conc=$conc<b=$b)"
      echo "RESULT $be $label b$b :: AGG=$agg tok/s  per_stream=$ps  ms/tok=$ms  conc=$conc$flag" | tee -a $SUM
    else
      err=$(grep -hiE "out of memory|CUDA error|EADDRINUSE|RuntimeError|AssertionError" $f 2>/dev/null | tail -1)
      echo "RESULT $be $label b$b :: FAILED: ${err:-see $f}" | tee -a $SUM
    fi
  done
done
echo "=== qwen8 throughput matrix done $(date +%H:%M:%S) ===" | tee -a $SUM
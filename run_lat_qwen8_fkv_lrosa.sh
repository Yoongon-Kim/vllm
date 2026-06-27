#!/usr/bin/env bash
# batch=1 decode-latency sweep, Qwen3-8B, budget(n_fac)=2048, YaRN.
# Backends: fkv (GPU0) + lrosa (GPU2) run in PARALLEL per context round.
# Contexts: 4k 8k 16k 32k 64k 128k.  LRoSA = deployed config (fp8_projk +
# radix + indexed_attend, cs_h=32) — same config we report accuracy for.
# Results -> /NHNHOME/jiwonsong/tmp/latbench/qwen8_<be>_<ctx>.txt + summary log.
set -u
PY=/NHNHOME/jiwonsong/miniconda3/envs/vllm/bin/python
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export HF_HOME=/NHNHOME/jiwonsong/hf_cache HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0

OUT=/NHNHOME/jiwonsong/tmp/latbench
SUM=$OUT/qwen8_fkv_lrosa_summary.log
MODEL=Qwen/Qwen3-8B
NFAC=2048; DECODE=128; BSZ=1; CSH=32
BASIS=/NHNHOME/jiwonsong/LRoSA-dev/bases/qwen3_8b/pca_d1_cs32_kv_head_qwen3_8b.pt

# 128k capped at 130816 so prefill+decode+16 <= YaRN window (32768*4=131072).
declare -A CTX=( [4k]=4096 [8k]=8192 [16k]=16384 [32k]=32768 [64k]=65536 [128k]=130816 )
CTX_ORDER=(4k 8k 16k 32k 64k 128k)
# backend -> GPU
declare -A GPU=( [fkv]=0 [lrosa]=2 )

echo "=== qwen8 fkv+lrosa latency sweep start $(date +%H:%M:%S) ===" | tee -a $SUM
for label in "${CTX_ORDER[@]}"; do
  plen=${CTX[$label]}
  echo "----- context $label ($plen) $(date +%H:%M:%S) -----" | tee -a $SUM
  for be in fkv lrosa; do
    g=${GPU[$be]}
    f=$OUT/qwen8_${be}_${label}.txt
    # fresh per-backend compile/inductor/triton cache -> no concurrent-proc race.
    CR=$HOME/.cache/vllm_lat_${be}
    TI=$TMPDIR/inductor_lat_${be}; TR=$TMPDIR/triton_lat_${be}
    rm -rf "$CR" "$TI" "$TR"; mkdir -p "$TI" "$TR"
    EXTRA=""
    if [ "$be" = "lrosa" ]; then EXTRA="--basis $BASIS --cs_h $CSH --fp8_projk"; fi
    CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT=$CR \
      TORCHINDUCTOR_CACHE_DIR=$TI TRITON_CACHE_DIR=$TR \
      $PY decode_latency_bench.py \
        --backend $be --model $MODEL --prefill_len $plen --decode_len $DECODE \
        --n_fac $NFAC --batch_size $BSZ --gpu_mem 0.90 $EXTRA \
        > $f 2>&1 &
  done
  wait
  for be in fkv lrosa; do
    f=$OUT/qwen8_${be}_${label}.txt
    line=$(grep -hE "DECODE_MS_PER_TOK" $f 2>/dev/null | tail -1)
    err=$(grep -hiE "out of memory|OutOfMemory|CUDA error|RuntimeError|AssertionError|Traceback" $f 2>/dev/null | tail -1)
    echo "RESULT $label $be :: ${line:-FAILED: ${err:-see $f}}" | tee -a $SUM
  done
done
echo "=== qwen8 fkv+lrosa latency sweep done $(date +%H:%M:%S) ===" | tee -a $SUM
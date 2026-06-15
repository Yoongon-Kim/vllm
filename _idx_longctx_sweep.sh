#!/bin/bash
# 64k/128k throughput sweep: FKV vs LRoSA OFF vs LRoSA ON, distributed across 8 GPUs.
# Each config pinned to one GPU with a unique VLLM_PORT (avoid distributed-port collision).
cd /NHNHOME/jiwonsong/vllm
source $HOME/miniforge3/etc/profile.d/conda.sh 2>/dev/null && conda activate vllm 2>/dev/null
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
M="Qwen/Qwen3-8B"
OUT=/tmp/idx_longctx
mkdir -p $OUT; rm -f $OUT/*.txt

# config: "ctx batch backend extra label"
CONFIGS=(
  "65536 8 fkv| |64k_b8_fkv"
  "65536 8 lrosa|--fp8_projk|64k_b8_off"
  "65536 8 lrosa|--fp8_projk --indexed_attend|64k_b8_on"
  "65536 12 fkv| |64k_b12_fkv"
  "65536 12 lrosa|--fp8_projk|64k_b12_off"
  "65536 12 lrosa|--fp8_projk --indexed_attend|64k_b12_on"
  "131072 4 fkv| |128k_b4_fkv"
  "131072 4 lrosa|--fp8_projk|128k_b4_off"
  "131072 4 lrosa|--fp8_projk --indexed_attend|128k_b4_on"
  "131072 6 fkv| |128k_b6_fkv"
  "131072 6 lrosa|--fp8_projk|128k_b6_off"
  "131072 6 lrosa|--fp8_projk --indexed_attend|128k_b6_on"
)

run_one(){
  local gpu=$1 idx=$2 cfg=$3
  IFS='|' read -r ctxbk extra label <<< "$cfg"
  read -r ctx batch backend <<< "$ctxbk"
  local port=$((23000 + idx))
  CUDA_VISIBLE_DEVICES=$gpu VLLM_PORT=$port python quest_latency_bench.py \
    --backend $backend --model "$M" --prefill_len $ctx --decode_len 64 \
    --n_fac 2048 --cs_h 32 $extra --batch_size $batch --gpu_mem 0.88 \
    > $OUT/$label.log 2>&1
  local line=$(grep -E "DECODE_MS" $OUT/$label.log | tail -1)
  if [ -z "$line" ]; then
    if grep -qiE "out of memory|OOM" $OUT/$label.log; then line="OOM"; else line="FAIL($(grep -ciE 'error|traceback' $OUT/$label.log))"; fi
  fi
  echo "$label : $line" >> $OUT/results.txt
}

# Wave 1: configs 0-7 on GPUs 0-7 (parallel, one job per GPU)
echo "=== WAVE 1 (8 configs on GPU 0-7) ==="
for i in 0 1 2 3 4 5 6 7; do
  ( run_one $i $i "${CONFIGS[$i]}" ) &
done
wait
echo "=== WAVE 1 done ==="
# Wave 2: configs 8-11 on GPUs 0-3 (parallel)
echo "=== WAVE 2 (4 configs on GPU 0-3) ==="
for i in 8 9 10 11; do
  gpu=$((i - 8))
  ( run_one $gpu $i "${CONFIGS[$i]}" ) &
done
wait
echo "=== ALL DONE ==="
sort $OUT/results.txt
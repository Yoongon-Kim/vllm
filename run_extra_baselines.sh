#!/usr/bin/env bash
# SqueezedAttention + SnapKV on the LRoSA-dev TRANSFORMERS stack (NOT vLLM).
# These two KV-eviction baselines are not ported to the vLLM backends; run them
# via pca's eval/longbench.py in the `pca` conda env. Both are single-point
# baselines at budget 256 (iso with n_fac=256), full 16 EN tasks, Qwen3 YaRN
# auto-enabled by the eval, --attn_impl sdpa (the pca env has no flash_attn),
# dynamic cache (these methods slice past_key_values directly), --resume.
#
# Runs the two methods in parallel on two GPUs. Usage:
#   GSQ=0 GSN=1 MODEL=Qwen/Qwen3-14B bash run_extra_baselines.sh
# Env: MODEL NSAMP MAXLEN GSQ (squeezed GPU) GSN (snapkv GPU)
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate pca
set -u
PYP="${PYP:-$HOME/miniforge3/envs/pca/bin/python}"
cd /NHNHOME/jiwonsong/LRoSA-dev
export HF_HUB_OFFLINE=1 TMPDIR=/NHNHOME/jiwonsong/tmp
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}

MODEL=${MODEL:-Qwen/Qwen3-14B}
NSAMP=${NSAMP:-200}; MAXLEN=${MAXLEN:-127500}
GSQ=${GSQ:-0}; GSN=${GSN:-1}
TAG=$("$PYP" -c "print('$MODEL'.split('/')[-1].lower().replace('.','_').replace('-','_'))")
OUT=results/longbench_v1
LOG=/NHNHOME/jiwonsong/tmp/extra_bl; mkdir -p "$LOG"
echo "=== Extra baselines (transformers stack): SqueezedAttn + SnapKV ==="
echo "=== model=$MODEL tag=$TAG nsamp=$NSAMP maxlen=$MAXLEN sdpa dynamic-cache resume ==="
echo "=== squeezed->GPU$GSQ  snap_kv->GPU$GSN  logs in $LOG ==="

CUDA_VISIBLE_DEVICES=$GSQ "$PYP" -m eval.longbench --base_model "$MODEL" \
  --mode squeezed --num_samples "$NSAMP" --max_input_len "$MAXLEN" \
  --attn_impl sdpa --cache_impl dynamic --resume \
  --squeezed_percent_clusters 6.25 --squeezed_percentile_budget 256 \
  --squeezed_obs_window 100 \
  --output_dir "$OUT" --run_name "squeezed_${TAG}" > "$LOG/squeezed.log" 2>&1 &
sq=$!

CUDA_VISIBLE_DEVICES=$GSN "$PYP" -m eval.longbench --base_model "$MODEL" \
  --mode snap_kv --num_samples "$NSAMP" --max_input_len "$MAXLEN" \
  --attn_impl sdpa --cache_impl dynamic --resume \
  --snap_max_capacity_prompt 256 --snap_window_size 32 --snap_kernel_size 5 \
  --snap_pooling avgpool \
  --output_dir "$OUT" --run_name "snap_kv_${TAG}" > "$LOG/snap_kv.log" 2>&1 &
sn=$!

# Heartbeat until both finish (full output is in the .log files). Progress =
# completed per-task .jsonl files in each run dir (eval writes one per task).
while :; do
  a=0; kill -0 $sq 2>/dev/null && a=$((a+1)); kill -0 $sn 2>/dev/null && a=$((a+1))
  sqd=$(ls "$OUT/squeezed_${TAG}"/*.jsonl 2>/dev/null | wc -l)
  snd=$(ls "$OUT/snap_kv_${TAG}"/*.jsonl 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] squeezed:${sqd}/16  snap_kv:${snd}/16  (alive $a/2)"
  [ "$a" -eq 0 ] && break
  sleep 120
done
echo "=== done; summaries: $OUT/{squeezed,snap_kv}_${TAG}/summary.json ==="

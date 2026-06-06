#!/usr/bin/env bash
# Fill the LongBench v1 table for ONE model across the 4 vLLM backends
# (fkv / lrosa / fasa / quest), one backend per GPU (4 GPUs) in parallel, then
# print a combined task x backend table.
#
# Usage:
#   MODEL=Qwen/Qwen3-14B bash run_longbench_table.sh                 # defaults
#   MODEL=Qwen/Qwen3-14B NSAMP=200 CS_H=32 bash run_longbench_table.sh
#   # Gemma 4 (head_dim>256): MODEL=google/gemma-4-26B-A4B-it CS_H=64 ...
#   # Qwen3-8B lives in HF_HOME=/NHNHOME/jiwonsong/hf_cache → export it first;
#   # Qwen3-14B / Gemma 4 live in the default ~/.cache/huggingface (offline).
#
# Env knobs (all optional): MODEL CS_H NFAC NSAMP MAXLEN GPUMEM TASKS GPUS
set -u
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate vllm
cd /NHNHOME/jiwonsong/vllm
export TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}     # models are pre-cached
# HF_HOME is inherited if you export it (Qwen3-8B); else the default cache is
# used (Qwen3-14B / Gemma 4).

MODEL=${MODEL:-Qwen/Qwen3-14B}
CS_H=${CS_H:-32}            # 64 for Gemma 4 (head_dim=512 full layers)
NFAC=${NFAC:-256}           # LongBench budget (paper default)
NSAMP=${NSAMP:-200}         # samples per task (0 = all)
MAXLEN=${MAXLEN:-127500}    # paper-faithful truncation cap
GPUMEM=${GPUMEM:-0.9}
TASKS=${TASKS:-all}         # or e.g. qasper,hotpotqa,gov_report
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"   # one GPU per backend
MODES=(fkv lrosa fasa quest)

TAG=$(python -c "import _bench_common as b; print(b.model_tag('$MODEL'))")
OUT=/NHNHOME/jiwonsong/tmp/lb_table/$TAG; mkdir -p "$OUT"
echo "=== LongBench table: $MODEL (tag=$TAG)  cs_h=$CS_H n_fac=$NFAC nsamp=$NSAMP maxlen=$MAXLEN ==="
echo "=== out: $OUT  $(date +%H:%M:%S) ==="

for i in "${!MODES[@]}"; do
  m=${MODES[$i]}; g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT="$HOME/.cache/lbt_${TAG}_${m}" \
    python longbench_vllm_eval.py --mode "$m" --model "$MODEL" --tasks "$TASKS" \
      --num_samples "$NSAMP" --n_fac "$NFAC" --cs_h "$CS_H" \
      --max_input_len "$MAXLEN" --gpu_mem "$GPUMEM" --output "$OUT/$m.json" \
      > "$OUT/$m.log" 2>&1 &
done
wait
echo "=== all backends done $(date +%H:%M:%S) ==="

python - "$OUT" <<'PY'
import glob, json, math, sys
d = sys.argv[1]
res = {}
for f in sorted(glob.glob(d + "/*.json")):
    j = json.load(open(f)); res[j["mode"]] = j
modes = [m for m in ["fkv", "lrosa", "fasa", "quest"] if m in res]
if not modes:
    print("no results — check the per-backend .log files in", d); raise SystemExit
tasks = list(res[modes[0]]["tasks"].keys())
print("\n=== LongBench v1 table  (per-task score; OVERALL = mean over tasks) ===")
print("%-22s" % "task" + "".join("%9s" % m for m in modes))
for t in tasks:
    print("%-22s" % t + "".join("%9.2f" % res[m]["tasks"].get(t, math.nan) for m in modes))
print("%-22s" % "OVERALL" + "".join("%9.2f" % res[m]["overall"] for m in modes))
# % of FKV (if present)
if "fkv" in res:
    base = res["fkv"]["overall"]
    print("%-22s" % "% of FKV" + "".join(
        "%9.1f" % (100 * res[m]["overall"] / base) for m in modes))
PY
echo "=== table done; per-backend json + logs in $OUT ==="

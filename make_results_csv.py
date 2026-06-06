"""Assemble a task-wise (horizontal) CSV of LongBench v1 results for Excel.

Rows = methods, columns = the 16 EN tasks (in eval order) + OVERALL.
Pulls from two sources, normalizing every score to the 0-100 scale:
  * vLLM sweep JSONs  /NHNHOME/jiwonsong/tmp/lb_sweep/<tag>/<label>.json
      {"tasks": {task: score_0_100}, "overall": ...}
  * transformers baselines (Squeezed/SnapKV)
      <PCA_REPO>/results/longbench_v1/<ModelName>/<run>_<tag>/summary.json
      {"per_task": {task: {"score": score_0_1}}, "overall_average": ...}  (x100)

Only methods with all 16 tasks (or status=="complete") are emitted; partial
runs are skipped (re-run this script once they finish to fill them in).

Usage:  python make_results_csv.py [tag]   (default tag=qwen3_14b)
"""
import csv
import glob
import json
import os
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen3_14b"
MODEL_NAME = {"qwen3_14b": "Qwen3-14B"}.get(TAG, TAG)
SWEEP = f"/NHNHOME/jiwonsong/tmp/lb_sweep/{TAG}"
PCA_RES = f"/NHNHOME/jiwonsong/LRoSA-dev/results/longbench_v1/{MODEL_NAME}"
OUT = f"{SWEEP}/results_{TAG}.csv"

# Preferred method row order (only those present are emitted).
ORDER = (
    ["fkv"]
    + [f"lrosa_cs{c}" for c in (4, 8, 16, 32)]
    + [f"loki_cs{c}" for c in (4, 8, 16, 32)]
    + [f"fasa_nt{n}" for n in (2, 4, 8, 16)]
    + ["quest", "squeezed", "snap_kv"]
)

methods = {}  # name -> {"tasks": {task: score_0_100}, "overall": float}

# 1) vLLM sweep JSONs (already 0-100).
for f in glob.glob(f"{SWEEP}/*.json"):
    label = os.path.basename(f)[:-5]
    j = json.load(open(f))
    t = j.get("tasks") or {}
    if len(t) >= 16:
        methods[label] = {"tasks": dict(t), "overall": j.get("overall")}

# 2) transformers Squeezed / SnapKV summaries (0-1 -> x100).
for m in ("squeezed", "snap_kv"):
    p = f"{PCA_RES}/{m}_{TAG}/summary.json"
    if not os.path.exists(p):
        continue
    j = json.load(open(p))
    pt = j.get("per_task") or {}
    if j.get("status") == "complete" and len(pt) >= 16:
        methods[m] = {
            "tasks": {k: v["score"] * 100.0 for k, v in pt.items()},
            "overall": (j.get("overall_average") or 0.0) * 100.0,
        }

if not methods:
    print("no complete methods found yet."); raise SystemExit

# Task column order: take it from fkv if present, else the longest method.
ref = methods.get("fkv") or max(methods.values(), key=lambda x: len(x["tasks"]))
TASKS = list(ref["tasks"].keys())

rows = [["method"] + TASKS + ["OVERALL"]]
for name in ORDER + [m for m in methods if m not in ORDER]:
    if name not in methods:
        continue
    d = methods[name]
    rows.append(
        [name]
        + [f"{d['tasks'].get(t, ''):.2f}" if isinstance(d["tasks"].get(t), (int, float)) else ""
           for t in TASKS]
        + [f"{d['overall']:.2f}" if isinstance(d["overall"], (int, float)) else ""]
    )

with open(OUT, "w", newline="") as fh:
    csv.writer(fh).writerows(rows)

# Also print to stdout (tab-separated is friendliest for direct paste).
print(f"# {len(methods)} methods x {len(TASKS)} tasks -> {OUT}\n")
for r in rows:
    print("\t".join(r))

"""Aggregate vLLM reasoning results (AIME25 / MATH-500 / GPQA) into one table.

Scans <PCA_REPO>/results/<eval>/<model>/<run>/summary.json (written by
reasoning_vllm_eval.py) and prints rows = (eval, backend, budget) with
pass@1, pass@k, mean_gen_len, frac_truncated, status. Tab-separated for Excel.

Usage:  python make_reasoning_csv.py [model_name]   (default Qwen3-8B)
"""
import glob
import json
import os
import sys

PCA = "/NHNHOME/jiwonsong/LRoSA-dev"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
EVALS = ["aime25", "math500", "gpqa"]
ORDER = {"fkv": 0, "lrosa": 1, "loki": 2, "quest": 3, "fasa": 4, "seer": 5}

rows = []
for ev in EVALS:
    for f in glob.glob(f"{PCA}/results/{ev}/{MODEL}/*/summary.json"):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get("engine") not in ("vllm", "seer"):  # skip stale transformers fmt
            continue
        mode = j.get("mode", "?")
        budget = "-" if mode == "fkv" else j.get("n_fac", "?")
        rows.append({
            "eval": ev, "backend": mode, "budget": budget,
            "pass@1": j.get("pass_at_1"), "pass@k": j.get("pass_at_k"),
            "gen_len": j.get("mean_gen_len"), "trunc": j.get("frac_truncated"),
            "runs": j.get("num_runs"), "n": j.get("num_samples"),
            "status": j.get("status", "?"),
        })

rows.sort(key=lambda r: (EVALS.index(r["eval"]), ORDER.get(r["backend"], 9),
                         -1 if r["budget"] == "-" else -int(r["budget"])))

hdr = ["eval", "backend", "budget", "pass@1", "pass@k", "gen_len", "trunc", "runs", "n", "status"]
print("\t".join(hdr))
out = [hdr]
for r in rows:
    def fmt(k):
        v = r[k]
        if v is None:
            return ""
        if k in ("pass@1", "pass@k", "trunc"):
            return f"{v:.4f}"
        if k == "gen_len":
            return f"{v:.0f}"
        return str(v)
    line = [fmt(k) for k in hdr]
    print("\t".join(line))
    out.append(line)

OUT = f"/NHNHOME/jiwonsong/tmp/reasoning_{MODEL}.csv"
import csv
with open(OUT, "w", newline="") as fh:
    csv.writer(fh).writerows(out)
print(f"\n# {len(rows)} runs -> {OUT}")

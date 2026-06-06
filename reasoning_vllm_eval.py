"""vLLM reasoning eval: AIME25 / MATH-500 / GPQA on the vLLM backends.

The reasoning counterpart of longbench_vllm_eval.py. Reasoning is short-prompt /
long-thinking-decode (the opposite of LongBench), so the sparse selection runs
over the GROWING decode trace. Backends: fkv / lrosa / loki / fasa / quest.

Scoring is byte-identical to pca's eval/{aime25,math500,gpqa}.py: it imports
their format_sample / extract_answer and pca's math_grader, so only the
generation engine differs (vLLM vs HF). Prompts use the chat template with
thinking ON (enable_thinking). pass@k via --num_runs. Writes pca-format
summaries + per-sample jsonl into the LRoSA results tree (results/<eval>/...).

Run:
  TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
  python reasoning_vllm_eval.py --eval gpqa --mode lrosa --model Qwen/Qwen3-8B \
      --cs_h 32 --n_fac 2048 --num_runs 1 --max_num_seqs 8
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from _bench_common import PCA_REPO, lrosa_basis_path, fasa_idom_path, yarn_overrides

sys.path.insert(0, PCA_REPO)
# Pure helpers reused verbatim from the transformers evals (identical scoring).
from eval.aime25 import format_sample as aime_format_sample  # noqa: E402
from eval.math500 import format_sample as math500_format_sample  # noqa: E402
from eval.gpqa import (  # noqa: E402
    format_sample as gpqa_format_sample,
    extract_answer as gpqa_extract_answer,
)
from eval.math_grader import (  # noqa: E402
    extract_answer as math_extract_answer,
    check_is_correct as math_check_is_correct,
)

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from datasets import load_dataset  # noqa: E402

# Per-model-family sampling (mirrors eval/aime25.py _SAMPLING_PRESETS).
SAMPLING_PRESETS = {
    "qwen3":    {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "gemma":    {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "gpt_oss":  {"temperature": 1.0, "top_p": 1.0,  "top_k": 0},
    "_default": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
}


def preset_for(model):
    n = model.lower()
    for key in ("qwen3", "gemma", "gpt_oss", "gpt-oss"):
        if key.replace("-", "_") in n.replace("-", "_"):
            return SAMPLING_PRESETS.get(key.replace("-", "_"), SAMPLING_PRESETS["_default"])
    return SAMPLING_PRESETS["_default"]


# ---- per-eval adapters: (load rows) + (row -> (prompt, gold, extra)) + grade ----

def load_rows(ev, a):
    if ev == "aime25":
        ds = load_dataset(a.aime_dataset, split=a.split or "train")
    elif ev == "math500":
        ds = load_dataset(a.math_dataset, split=a.split or "test")
    elif ev == "gpqa":
        ds = load_dataset(a.gpqa_dataset, a.gpqa_config, split=a.split or "train")
    else:
        raise ValueError(ev)
    rows = list(ds)
    if a.num_samples:
        rows = rows[: a.num_samples]
    return rows


def build_prompt_and_gold(ev, row, idx, seed):
    """Returns (prompt_text, gold) where gold is the answer / letter."""
    if ev == "aime25":
        p, gold = aime_format_sample(row)
        return p, gold
    if ev == "math500":
        p, gold = math500_format_sample(row)
        return p, gold
    if ev == "gpqa":
        # Stable per-(question, seed) option shuffle, like eval/gpqa.py.
        rng = random.Random(seed * 100000 + idx)
        p, gold_letter, _opts = gpqa_format_sample(row, rng)
        return p, gold_letter


def grade(ev, response, gold):
    if ev == "gpqa":
        return gpqa_extract_answer(response) == gold
    # aime25 / math500: boxed extraction + math_equal
    return bool(math_check_is_correct(math_extract_answer(response), gold))


def build_llm(a):
    kw = dict(
        model=a.model,
        max_model_len=a.max_input_len + a.max_new_tokens + 16,
        gpu_memory_utilization=a.gpu_mem,
        enforce_eager=a.eager,
        enable_prefix_caching=False,
    )
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    ov = yarn_overrides(a.model)
    if ov:
        kw["hf_overrides"] = ov
    if a.mode in ("lrosa", "loki"):
        variant = "loki" if a.mode == "loki" else "d1"
        basis = a.basis or lrosa_basis_path(a.model, cs_h=a.cs_h, variant=variant)
        kw["kv_cache_dtype"] = "lrosa"
        ac = {"backend": "LROSA", "lrosa_basis_path": basis,
              "lrosa_n_fac": a.n_fac, "lrosa_cs_h": a.cs_h,
              "lrosa_use_radix_topk": True}
        if a.fp8_projk:
            ac["lrosa_fp8_projk"] = True
        kw["attention_config"] = ac
    elif a.mode == "fasa":
        kw["kv_cache_dtype"] = "fasa"
        kw["attention_config"] = {"backend": "LROSA",
                                  "lrosa_basis_path": a.basis or fasa_idom_path(a.model),
                                  "lrosa_n_fac": a.n_fac, "lrosa_n_tip": a.n_tip,
                                  "lrosa_use_radix_topk": True}
    elif a.mode == "quest":
        kw["kv_cache_dtype"] = "quest"
        kw["attention_config"] = {"backend": "QUEST", "quest_token_budget": a.n_fac}
    # fkv: dense default.
    return LLM(**kw)


def tokenize(tok, prompt_text, model):
    msgs = [{"role": "user", "content": prompt_text}]
    kw = dict(tokenize=True, add_generation_prompt=True)
    if "enable_thinking" in (tok.chat_template or ""):
        kw["enable_thinking"] = True
    ids = tok.apply_chat_template(msgs, **kw)
    return ids if isinstance(ids, list) else ids["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=["aime25", "math500", "gpqa"])
    ap.add_argument("--mode", default="lrosa",
                    choices=["fkv", "lrosa", "loki", "fasa", "quest"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--num_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--num_runs", type=int, default=1, help="attempts/problem (pass@1 avg)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="skip (run, idx) attempts already in predictions.jsonl "
                         "(default on; --no-resume to start fresh).")
    ap.add_argument("--max_new_tokens", type=int, default=38912)
    ap.add_argument("--max_input_len", type=int, default=4096)
    ap.add_argument("--n_fac", type=int, default=2048)
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16)
    ap.add_argument("--basis", default=None)
    # FP8 score is the DEFAULT for LRoSA/Loki (lossless vs bf16, deployment
    # config); --no-fp8_projk for bf16. Ignored for fkv/fasa/quest and
    # auto-gated off for head_size>256 (e.g. Gemma 4 -> effectively bf16).
    ap.add_argument("--fp8_projk", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None)
    ap.add_argument("--aime_dataset", default="yentinglin/aime_2025")
    ap.add_argument("--math_dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--gpqa_dataset", default="Idavidrein/gpqa")
    ap.add_argument("--gpqa_config", default="gpqa_diamond")
    ap.add_argument("--results_dir", default=os.path.join(PCA_REPO, "results"))
    ap.add_argument("--run_name", default=None)
    a = ap.parse_args()

    pr = preset_for(a.model)
    if a.temperature is None:
        a.temperature = pr["temperature"]
    if a.top_p is None:
        a.top_p = pr["top_p"]
    if a.top_k is None:
        a.top_k = pr["top_k"]
    if a.run_name is None:
        a.run_name = f"{a.mode}" + (f"_cs{a.cs_h}" if a.mode in ("lrosa", "loki") else "") \
                     + ("_fp8" if a.fp8_projk else "")

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = load_rows(a.eval, a)
    llm = build_llm(a)

    run_dir = None
    if a.results_dir:
        run_dir = os.path.join(a.results_dir, a.eval, a.model.split("/")[-1], a.run_name)
        os.makedirs(run_dir, exist_ok=True)

    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p,
                        top_k=(a.top_k if a.top_k and a.top_k > 0 else -1),
                        max_tokens=a.max_new_tokens)

    n = len(rows)
    pred_path = os.path.join(run_dir, "predictions.jsonl") if run_dir else None
    summ_path = os.path.join(run_dir, "summary.json") if run_dir else None

    # ---- Resume: skip (run, idx) attempts already in predictions.jsonl ----
    # predictions.jsonl is appended + flushed per attempt, so an interrupted
    # run resumes from exactly where it stopped. Guard against config drift:
    # if an existing summary's key knobs differ, start fresh (don't mix).
    recs, done = [], set()
    fresh = True
    if pred_path and a.resume and os.path.exists(pred_path):
        ok_cfg = True
        if os.path.exists(summ_path):
            try:
                old = json.load(open(summ_path))
                for k, v in [("base_model", a.model), ("eval", a.eval),
                             ("mode", a.mode), ("fp8_projk", bool(a.fp8_projk)),
                             ("max_new_tokens", a.max_new_tokens),
                             ("n_fac", a.n_fac), ("cs_h", a.cs_h)]:
                    if old.get(k) != v:
                        print(f"  resume: config drift ({k}: {old.get(k)} != {v}) "
                              f"-> starting FRESH", flush=True)
                        ok_cfg = False
                        break
            except Exception:
                pass
        if ok_cfg:
            for line in open(pred_path):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["idx"] < n:
                    recs.append(r)
                    done.add((r["run"], r["idx"]))
            fresh = False
            print(f"  resume: {len(done)} attempts already done in {pred_path}", flush=True)

    def aggregate(status):
        sc = [0] * n
        gl = [[] for _ in range(n)]
        rcorr, rtot = {}, {}
        for r in recs:
            i = r["idx"]
            sc[i] += int(bool(r.get("correct")))
            gl[i].append(r.get("gen_len", 0))
            rcorr[r["run"]] = rcorr.get(r["run"], 0) + int(bool(r.get("correct")))
            rtot[r["run"]] = rtot.get(r["run"], 0) + 1
        attempts = sum(len(g) for g in gl)
        p1 = sum(sc) / attempts if attempts else 0.0
        pk = sum(1 for c in sc if c > 0) / n if n else 0.0
        alll = [x for sub in gl for x in sub]
        mgl = sum(alll) / len(alll) if alll else 0.0
        per_run = [rcorr[k] / rtot[k] for k in sorted(rtot)]
        per_problem = [
            {"idx": i, "solved_count": sc[i], "n_runs": len(gl[i]),
             "pass_at_1": (sc[i] / len(gl[i]) if gl[i] else 0.0),
             "gen_lens": gl[i],
             "mean_gen_len": (sum(gl[i]) / len(gl[i]) if gl[i] else 0.0)}
            for i in range(n)]
        summ = {
            "run_name": a.run_name, "eval": a.eval, "base_model": a.model,
            "mode": a.mode, "engine": "vllm", "cs_h": a.cs_h, "n_fac": a.n_fac,
            "n_tip": a.n_tip, "fp8_projk": bool(a.fp8_projk),
            "max_new_tokens": a.max_new_tokens, "num_samples": n,
            "num_runs": a.num_runs, "completed_attempts": attempts,
            "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
            "status": status, "run_accuracies": per_run, "pass_at_1": p1,
            "pass_at_k": pk, "avg_accuracy": p1, "solved_count": sc,
            "mean_gen_len": mgl, "max_gen_len": max(alll) if alll else 0,
            "frac_truncated": (sum(1 for x in alll if x >= a.max_new_tokens) / len(alll)
                               if alll else 0.0),
            "per_problem": per_problem,
        }
        if summ_path:
            with open(summ_path, "w") as f:
                json.dump(summ, f, indent=2)
                f.write("\n")
        return p1, pk, mgl

    pf = open(pred_path, "a" if not fresh else "w") if pred_path else None
    for run_id in range(a.num_runs):
        pending = [i for i in range(n) if (run_id, i) not in done]
        if not pending:
            print(f"[REASON] {a.eval} {a.mode} run{run_id}: all {n} done (resumed)", flush=True)
            continue
        sp.seed = a.seed + run_id
        prompts, golds = [], []
        for i in pending:
            ptext, gold = build_prompt_and_gold(a.eval, rows[i], i, a.seed + run_id)
            golds.append(gold)
            prompts.append(TokensPrompt(prompt_token_ids=tokenize(tok, ptext, a.model)))
        outs = llm.generate(prompts, sp, use_tqdm=True)
        c = 0
        for i, o, gold in zip(pending, outs, golds):
            glen = len(o.outputs[0].token_ids)
            ok = grade(a.eval, o.outputs[0].text, gold)
            c += int(ok)
            rec = {"run": run_id, "idx": i, "gold": gold, "gen_len": glen,
                   "correct": ok, "truncated": glen >= a.max_new_tokens}
            recs.append(rec)
            done.add((run_id, i))
            if pf:
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pf.flush()
        print(f"[REASON] {a.eval} {a.mode} run{run_id} +{len(pending)} acc={c / len(pending):.4f}",
              flush=True)
        if run_dir:
            aggregate("partial")
    if pf:
        pf.close()

    pass_at_1, pass_at_k, mean_gen_len = aggregate("complete")
    print(f"\n=== {a.eval} mode={a.mode} model={a.model} cs_h={a.cs_h} n_fac={a.n_fac} "
          f"runs={a.num_runs} fp8={bool(a.fp8_projk)} ===")
    print(f"  pass@1 = {pass_at_1:.4f}  (avg over {a.num_runs} attempts/problem, n={n})")
    print(f"  pass@{a.num_runs} = {pass_at_k:.4f}  (solved by >=1 attempt)")
    print(f"  mean_gen_len = {mean_gen_len:.0f} tokens")
    if run_dir:
        print(f"  saved -> {run_dir}")


if __name__ == "__main__":
    main()

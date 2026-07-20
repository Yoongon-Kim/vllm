"""Phase 3b smoke: run QUEST decode in the real vLLM engine (cudagraph),
FlashInfer per-head attend (QUEST_FI_ATTEND=1) vs the custom Triton kernel (=0),
greedy generation on a >n_fac prompt (sparse regime). Prints the generated token
ids so an external A/B compares them (must match up to bf16-kernel noise)."""
import os
import sys
os.environ.setdefault("HF_HOME", "/NHNHOME/jiwonsong/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
sys.path.insert(0, "/NHNHOME/jiwonsong/vllm")

import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from decode_latency_bench import build_llm  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
PREFILL = int(os.environ.get("SMOKE_PREFILL", "8192"))   # > n_fac -> sparse regime
NFAC = int(os.environ.get("SMOKE_NFAC", "2048"))
NEW = int(os.environ.get("SMOKE_NEW", "48"))
fi = os.environ.get("QUEST_FI_ATTEND", "1")

llm = build_llm("quest", PREFILL, NEW, NFAC, 0.85, 1, model=MODEL)
torch.manual_seed(0)
# deterministic pseudo-prompt token ids in a safe range
toks = [((i * 7919) % 30000) + 5 for i in range(PREFILL)]
sp = SamplingParams(temperature=0.0, max_tokens=NEW)
out = llm.generate([TokensPrompt(prompt_token_ids=toks)], sp, use_tqdm=False)
ids = list(out[0].outputs[0].token_ids)
print(f"RESULT fi={fi} n={len(ids)} tokens={ids}")

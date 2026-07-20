"""QUEST throughput grid over 8 GPUs. One decode_latency_bench subprocess per
cell (model x ctx x batch), 8 concurrent workers (one per GPU). GRID_FI=1 -> the
new FlashInfer per-head attend (default); GRID_FI=0 -> the old custom Triton
kernel (for the before/after delta). Writes an incremental CSV.
"""
import csv
import os
import queue
import re
import signal
import subprocess
import threading

MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "gemma-4-26b": "google/gemma-4-26b-a4b-it"}
CTX = [8192, 16384, 32768, 65536, 131072]
BATCH = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# per-model per-ctx max batch (user-set caps; drop the large-batch long-ctx cells
# that OOM / aren't deployment-representative for an 8B-vs-26B comparison).
MAXB = {
    "gemma-4-26b": {8192: 256, 16384: 256, 32768: 64, 65536: 32, 131072: 16},
    "qwen3-8b":    {8192: 256, 16384: 256, 32768: 32, 65536: 16, 131072: 8},
}
NFAC = 2048
NGPU = 8
PY = "/NHNHOME/jiwonsong/miniconda3/envs/vllm/bin/python"
BENCH = "/NHNHOME/jiwonsong/vllm/decode_latency_bench.py"
FI = os.environ.get("GRID_FI", "1")
OUT = f"/NHNHOME/jiwonsong/OptProd_results/quest_throughput_fi{FI}.csv"
PAT = re.compile(
    r"DECODE_MS_PER_TOK=([\d.]+)\s+PER_STREAM_TOK_S=([\d.]+)\s+AGG_TOK_S=([\d.]+)")

RESUME = os.environ.get("GRID_RESUME", "0") == "1"
results = {}
if RESUME and os.path.exists(OUT):
    # keep already-ok cells that are still WITHIN the (possibly tightened) caps
    with open(OUT) as f:
        for row in csv.reader(f):
            if row and row[0] != "model":
                mk, fi, ctx, b, dms, tps, st = row
                st = st.strip()
                if st == "ok" and int(b) <= MAXB.get(mk, {}).get(int(ctx), 0):
                    results[(mk, int(ctx), int(b))] = (mk, fi, int(ctx), int(b),
                                                       dms, tps, st)

cells = []
port = 47000
for mk, mv in MODELS.items():
    for ctx in CTX:
        for b in BATCH:
            if b <= MAXB[mk][ctx] and (mk, ctx, b) not in results:
                cells.append((mk, mv, ctx, b, port))
                port += 5                              # unique port per cell (no reuse race)

work = queue.Queue()
for c in cells:
    work.put(c)
lock = threading.Lock()
print(f"cells to run: {len(cells)} (resume kept {len(results)})", flush=True)


def flush():
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fi", "ctx", "batch", "decode_ms", "agg_tok_s", "status"])
        for k in sorted(results):
            w.writerow(results[k])


def worker(gpu):
    while True:
        try:
            mk, mv, ctx, b, port = work.get_nowait()
        except queue.Empty:
            return
        prefill = ctx
        if mk.startswith("qwen") and ctx >= 131072:
            prefill = 130816                       # YaRN cap headroom for decode
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   QUEST_FI_ATTEND=FI, VLLM_PORT=str(port))
        cmd = [PY, BENCH, "--backend", "quest", "--model", mv,
               "--prefill_len", str(prefill), "--decode_len", "128",
               "--n_fac", str(NFAC), "--batch_size", str(b), "--gpu_mem", "0.9"]
        status, dms, tps = "err", "", ""
        out = ""
        # start_new_session -> own process group so we can kill the WHOLE tree
        # (vLLM EngineCore/Workers) on timeout; else they orphan and hog the GPU,
        # OOM-cascading every later cell on this GPU.
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=1200)
            m = PAT.search(out)
            if m:
                status, dms, tps = "ok", float(m.group(1)), float(m.group(3))
            elif ("out of memory" in out.lower() or "outofmemory" in out.lower()):
                status = "OOM"
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            # SIGKILL frees GPU memory asynchronously; wait until this GPU
            # actually settles before the next cell starts (else the dying
            # process's leftover memory OOMs the next cell — false OOMs).
            for _ in range(40):
                try:
                    q = subprocess.run(
                        ["nvidia-smi", "-i", str(gpu),
                         "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=15)
                    if int(q.stdout.strip().splitlines()[0]) < 2000:
                        break
                except Exception:
                    break
                __import__("time").sleep(2)
        if status != "ok":  # save output for debugging non-ok cells
            with open(f"/NHNHOME/jiwonsong/tmp/gridcell_fi{FI}_{mk}_{ctx}_{b}.log",
                      "w") as lf:
                lf.write(out[-6000:])
        with lock:
            results[(mk, ctx, b)] = (mk, FI, ctx, b, dms, tps, status)
            flush()
            print(f"[gpu{gpu}] {mk} ctx{ctx} b{b} -> {status} tok/s={tps}", flush=True)
        work.task_done()


threads = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"GRID DONE fi={FI}: {len(results)}/{len(cells)} cells -> {OUT}", flush=True)

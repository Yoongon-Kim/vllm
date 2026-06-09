#!/usr/bin/env bash
# resume_reasoning.sh — relaunch the paused reasoning runs. ALL resume from saved
# predictions (gpqa_seer: per-(run,idx) chunk resume; reasoning_vllm_eval: skips
# fully-completed runs). Run this when GPUs are FREE (after the MLA dev work).
#
# Paused-at snapshot (2026-06-08):
#   seer GPQA 1024 : 115/1584 attempts saved
#   MATH-500 nf512 : lrosa 6/8 runs, quest 4/8, fasa 3/8 saved
#
# Layout: GPU3 seer-1024 ; GPU0/1/2 MATH-500 nf512 lrosa/quest/fasa ;
#         seer-512 queued via waiter onto the first GPU that frees.
V=/NHNHOME/jiwonsong/vllm; L=/NHNHOME/jiwonsong/tmp
nohup bash "$V/run_gpqa_seer.sh" 1024 3        > "$L/resume_seer1024.log"     2>&1 & echo "seer-1024  -> GPU3 (pid $!)"
nohup bash "$V/run_reason.sh" math500 lrosa 512 0 > "$L/resume_math_lrosa.log" 2>&1 & echo "math512 lrosa -> GPU0 (pid $!)"
nohup bash "$V/run_reason.sh" math500 quest 512 1 > "$L/resume_math_quest.log" 2>&1 & echo "math512 quest -> GPU1 (pid $!)"
nohup bash "$V/run_reason.sh" math500 fasa  512 2 > "$L/resume_math_fasa.log"  2>&1 & echo "math512 fasa  -> GPU2 (pid $!)"
nohup bash "$V/seer512_waiter.sh"              > "$L/resume_seer512_waiter.log" 2>&1 & echo "seer-512 waiter (pid $!)"
echo "resumed. monitor: tail -f $L/resume_*.log"

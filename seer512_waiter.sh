#!/usr/bin/env bash
# Wait until a non-GPU3 GPU frees (<20GB used), then launch seer GPQA budget-512
# on it. Resume-safe (gpqa_seer skips done attempts). Bracket-regex avoids
# matching this waiter's own cmdline.
SCR=/NHNHOME/jiwonsong/vllm/run_gpqa_seer.sh
for i in $(seq 1 600); do
  if pgrep -f '[e]val\.gpqa_seer.*budget 512' >/dev/null 2>&1; then exit 0; fi
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '($1+0)!=3 && ($2+0)<20000 {print $1; exit}')
  if [ -n "$free" ]; then
    nohup bash "$SCR" 512 "$free" >/dev/null 2>&1 &
    exit 0
  fi
  sleep 120
done

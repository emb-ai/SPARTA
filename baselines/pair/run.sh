#!/usr/bin/env bash

PYTHON="python"
WAIT_TIME=1800
SESSION_BASE="lisa_cycle"

SAVE_PATHS=(
  "lisa++"
  "lisa-13b-v1-exp"
  "lisa-7b-v1"
  "lisa-7b-v1-exp"
  "lisa-13b-v1"
  "gsva-13b-llama2-ft-res"
)

idx=0
for SAVE in "${SAVE_PATHS[@]}"; do
  ((idx++))
  SESSION="${SESSION_BASE}_${idx}"

  echo "=== Cycle $idx / ${#SAVE_PATHS[@]}  (SAVE_PATH=$SAVE) ==="

  # launch seg_service in its own tmux session
  SESSION_SEG="${SESSION}_seg"
  
  # tmux new-session -d -s "$SESSION_SEG" -n seg_service \
  #       "conda run --live-stream -n antonov-sonar \
  #       python services/seg_service.py \
  #       --port 8000 --log-level debug \
  #       segmod@_global_=$SAVE attacker=llm dataset=reason_test"
  
  tmux new-session -d -s "$SESSION_SEG" -n seg_service \
        "conda run --live-stream -n antonov-sonar \                     
        python services/seg_service.py \
        --port 8000 \
        --log-level debug \
        segmod@_global_=$SAVE attacker=llm dataset=llmseg_val"


  # launch llm_service in its own tmux session
  SESSION_LLM="${SESSION}_llm"
  tmux new-session -d -s "$SESSION_LLM" -n llm_service \
        "conda run --live-stream -n antonov-qwen3 \
        python3 services/llm_service.py --port 8001"

  echo "  seg_service.py and llm_service.py launched in sessions $SESSION_SEG and $SESSION_LLM"

  echo "  waiting $((WAIT_TIME/60)) minutes for services to warm up…"
  sleep "$WAIT_TIME"

  echo "  running PAIR.py …"
  SAVE_PREFIX="adv_search_results_${SAVE}"
  # SAVE_PATH="$SAVE_PREFIX" "$PYTHON" PAIR.py --dataset reason_test --questions_file processed_texts.txt # --max_samples 300 
  SAVE_PATH="$SAVE_PREFIX" "$PYTHON" PAIR.py --max_samples 300 
  echo "  PAIR.py finished"

  tmux kill-session -t "$SESSION_SEG"
  tmux kill-session -t "$SESSION_LLM"
  echo "  tmux sessions $SESSION_SEG and $SESSION_LLM killed (services stopped)"
  echo
done

echo "=== All cycles completed ==="
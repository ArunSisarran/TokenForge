#!/usr/bin/env bash
# TokenForge Phase 2, pieces 2-5: vLLM scheduler limits, prefix caching,
# Phase 1 wins under load, and SGLang, run back to back.
#
# Run on the pod from the repo root, in tmux, with /workspace/venv active:
#     bash bench/phase2.sh            # all pieces
#     bash bench/phase2.sh 3 4        # only pieces 3 and 4
#     M=/workspace/models/Qwen3-8B-FP8-Dynamic TAG=fp8 bash bench/phase2.sh
# Server logs and /metrics dumps go to /workspace/logs/.

M=${M:-/workspace/models/Qwen3-8B-W4A16-G128}
TAG=${TAG:-int4}
LOGS=${LOGS:-/workspace/logs}
EAGLE='{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'
LOAD=(--num-requests 32 --min-rounds 4 --ignore-eos --max-tokens-cap 256 --gpu-cost-per-hour 0.70)
mkdir -p $LOGS

wait_ready() {  # pid, health path
  until curl -sf localhost:8000$2 > /dev/null; do kill -0 $1 2>/dev/null || return 1; sleep 5; done
}

stop() {  # pid
  kill $1 2>/dev/null; wait $1 2>/dev/null
  # SGLang and vLLM run engine subprocesses that must release the GPU before the next start.
  pkill -f "vllm serve|sglang.launch_server" 2>/dev/null; sleep 5
}

bench() {  # label, workload, concurrency levels, extra loadgen flags...
  local label=$1 workload=$2 conc=$3; shift 3
  python bench/loadgen.py --workload bench/workloads/$workload.jsonl --model $M \
    --concurrency $conc "${LOAD[@]}" --label $label "$@"
  curl -s localhost:8000/metrics | grep -E "^vllm:(num_preemptions|prefix_cache|spec_decode)" \
    > $LOGS/$label-$workload.metrics
}

run() {  # label, workload, concurrency levels, vllm flags...
  local label=$1 workload=$2 conc=$3; shift 3
  echo "=== $label ($workload)"
  vllm serve $M --gpu-memory-utilization 0.85 \
    --default-chat-template-kwargs '{"enable_thinking": false}' "$@" \
    > $LOGS/$label-$workload.log 2>&1 &
  local pid=$!
  if wait_ready $pid /health; then
    bench $label $workload "$conc"
  else
    echo "!!! $label failed to start, see $LOGS/$label-$workload.log"
  fi
  stop $pid
}

piece2() {  # scheduler limits: batch cap on chat, per-step token budget on long prompts
  for s in 8 16 32; do
    run vllm-$TAG-seqs$s-l4 chat "32 64" --max-model-len 4096 --no-enable-prefix-caching \
      --max-num-seqs $s
  done
  for b in 512 2048 8192; do
    run vllm-$TAG-mbt$b-l4 long_context "8" --max-model-len 16384 --no-enable-prefix-caching \
      --max-num-seqs 64 --max-num-batched-tokens $b
  done
}

piece3() {  # prefix caching on the shared ~2k-token system prompt
  run vllm-$TAG-nocache-l4 shared_prefix "1 4 16 32" --max-model-len 4096 --no-enable-prefix-caching
  run vllm-$TAG-prefixcache-l4 shared_prefix "1 4 16 32" --max-model-len 4096 --enable-prefix-caching
}

piece4() {  # Phase 1 wins under load: FP8 KV cache and EAGLE-3
  run vllm-$TAG-kvfp8-batch-l4 chat "1 2 4 8 16 32 64" --max-model-len 4096 \
    --no-enable-prefix-caching --kv-cache-dtype fp8
  run vllm-$TAG-eagle3k3-batch-l4 chat "1 2 4 8 16 32 64" --max-model-len 4096 \
    --no-enable-prefix-caching --speculative-config "$EAGLE"
}

piece5() {  # SGLang on the piece 1 workload, in its own venv
  local py=/workspace/venv-sglang/bin/python
  [ -x $py ] || { python -m venv /workspace/venv-sglang && $py -m pip install "sglang[all]"; }
  $py -c "import sglang; print('sglang', sglang.__version__)" | tee $LOGS/sglang-version.txt
  echo "=== sglang-$TAG-batch-l4 (chat)"
  $py -m sglang.launch_server --model-path $M --port 8000 --context-length 4096 \
    --mem-fraction-static 0.85 --disable-radix-cache > $LOGS/sglang-$TAG-batch-l4-chat.log 2>&1 &
  local pid=$!
  if wait_ready $pid /health_generate; then
    bench sglang-$TAG-batch-l4 chat "1 2 4 8 16 32 64" \
      --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
  else
    echo "!!! SGLang failed to start, see $LOGS/sglang-$TAG-batch-l4-chat.log"
  fi
  stop $pid
}

for p in ${*:-2 3 4 5}; do "piece$p"; done
echo "Done. Results in bench/results/, logs and metrics in $LOGS/"

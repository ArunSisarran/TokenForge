#!/usr/bin/env bash
# TokenForge Phase 5, piece 2: restart the gateway Deployment twice while loadgen keeps
# 32 users busy, so every failed request in the result file comes from pod replacement.
#
# Run from the repo root with the venv active, against the kind cluster:
#     KEY=<gateway key> LABEL=k8s-rollout-<variant> bash bench/rollout.sh
set -euo pipefail
: "${KEY:?set KEY to a gateway API key}" "${LABEL:?set LABEL}"
ulimit -n 4096

python bench/loadgen.py --url http://localhost:8080 --workload bench/workloads/chat.jsonl \
  --model qwen3-8b --concurrency 32 --num-requests 500 --ignore-eos --max-tokens-cap 256 \
  --api-key "$KEY" --label "$LABEL" &
load=$!

sleep 10
for i in 1 2; do
  kubectl rollout restart deploy/tf-gateway
  kubectl rollout status deploy/tf-gateway --timeout 5m
  sleep 5
done
wait $load

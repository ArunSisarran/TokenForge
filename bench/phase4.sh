#!/usr/bin/env bash
# TokenForge Phase 4, piece 4: the compose stack on this laptop in front of vLLM on a
# RunPod pod running the official vllm/vllm-openai image. Benchmarks the pod directly and
# through the containerized gateway, then scans the vLLM image with Trivy.
#
# Run from the repo root with the venv active, once the pod is deployed:
#     POD=<public-ip>:<external-port> VLLM_KEY=<the pod's VLLM_API_KEY> bash bench/phase4.sh
set -euo pipefail
: "${POD:?set POD=<public-ip>:<external-port>}" "${VLLM_KEY:?set VLLM_KEY}"
CONC=${CONC:-1 2 4 8 16 32 64}
VLLM_IMAGE=vllm/vllm-openai:v0.34.0
LOAD=(--workload bench/workloads/chat.jsonl --model qwen3-8b --concurrency $CONC
      --num-requests 32 --min-rounds 4 --ignore-eos --max-tokens-cap 256 --gpu-cost-per-hour 0.70
      --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}')
ulimit -n 4096

echo "=== waiting for vLLM at $POD (model load takes a few minutes)"
until curl -sf "http://$POD/health" > /dev/null; do sleep 10; done
echo "vLLM $(curl -s "http://$POD/version")"
code=$(curl -s -o /dev/null -w "%{http_code}" "http://$POD/v1/models")
[ "$code" = 401 ] || { echo "!!! the pod answered $code without a key: it is open to the internet, stop it"; exit 1; }

echo "=== compose stack: gateway + Redis, engine = the pod"
docker compose --profile fake down
UPSTREAM_URL="http://$POD" UPSTREAM_API_KEY="$VLLM_KEY" docker compose up -d --build --wait
KEY=$(docker compose exec -T gateway python -m gateway.keys create phase4 --rpm 1000000 --tpm 1000000000)

echo "=== direct to the pod"
python bench/loadgen.py --url "http://$POD" --api-key "$VLLM_KEY" --label vllm-int4-kvfp8-pod-direct-l4 "${LOAD[@]}"
echo "=== through the gateway container"
python bench/loadgen.py --url http://localhost:8080 --api-key "$KEY" --label gateway-docker-int4-kvfp8-l4 "${LOAD[@]}"

echo
echo ">>> GPU work is done: terminate the pod now. Trivy below runs on this laptop (~10 GB download)."
trivy image --scanners vuln --platform linux/amd64 --timeout 60m "$VLLM_IMAGE" \
  > bench/results/trivy-vllm-openai-v0.34.0.txt
sed -n '/Report Summary/,/Legend/p' bench/results/trivy-vllm-openai-v0.34.0.txt | grep -Ev "│ +0 +│"
echo "Done. Results in bench/results/."

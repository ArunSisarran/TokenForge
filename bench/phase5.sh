#!/usr/bin/env bash
# TokenForge Phase 5, piece 3: the kind cluster in front of real vLLM on a RunPod pod.
# Creates the pod, points the Helm release at it, benchmarks direct and through the
# cluster alternately (direct, cluster, direct, cluster), then deletes the pod and
# switches the release back to the simulator. The pod is deleted on any exit.
#
# Run from the repo root with the venv active and the kind cluster up (piece 1):
#     RUNPOD_API_KEY=<key> VOL=<network volume id> DC=<its dataCenter> GPU_ID="<L4 id>" \
#         bash bench/phase5.sh
set -euo pipefail
: "${RUNPOD_API_KEY:?}" "${VOL:?}" "${DC:?}" "${GPU_ID:?}"
API=${RUNPOD_API:-https://api.runpod.io/v2}
CONC=${CONC:-1 2 4 8 16 32 64}
H="Authorization: Bearer $RUNPOD_API_KEY"
VLLM_KEY=$(openssl rand -hex 24)
LOAD=(--workload bench/workloads/chat.jsonl --model qwen3-8b --concurrency $CONC
      --num-requests 32 --min-rounds 4 --ignore-eos --max-tokens-cap 256 --gpu-cost-per-hour 0.70
      --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}')
ulimit -n 4096

body=$(VLLM_KEY=$VLLM_KEY python3 -c '
import json, os
print(json.dumps({
    "name": "tokenforge-phase5", "image": "vllm/vllm-openai:v0.34.0", "cloud": "SECURE",
    "gpu": {"id": os.environ["GPU_ID"], "count": 1}, "dataCenterIds": [os.environ["DC"]],
    "mounts": {"network": [{"volumeId": os.environ["VOL"], "path": "/workspace"}]},
    "ports": ["8000/tcp"], "disk": 20, "env": {"VLLM_API_KEY": os.environ["VLLM_KEY"]},
    "args": "/workspace/models/Qwen3-8B-W4A16-G128 --served-model-name qwen3-8b --host 0.0.0.0"
            " --port 8000 --gpu-memory-utilization 0.85 --max-model-len 4096"
            " --no-enable-prefix-caching --kv-cache-dtype fp8"}))')
POD_ID=$(curl -sf -X POST "$API/pods" -H "$H" -H 'Content-Type: application/json' -d "$body" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')
echo "=== created pod $POD_ID (billing has started)"

cleanup() {
  curl -s -o /dev/null -w "=== pod $POD_ID deleted: HTTP %{http_code} (expect 204)\n" -X DELETE -H "$H" "$API/pods/$POD_ID"
  helm upgrade tf deploy/helm/tokenforge --reset-values --wait --timeout 6m > /dev/null && echo "=== release back on the simulator"
}
trap cleanup EXIT
trap 'exit 130' INT TERM  # Ctrl-C or kill becomes a normal exit, so cleanup always runs

echo "=== waiting for the pod's public port (max 15 min)"
for i in $(seq 90); do
  # Prints ip:port once mapped, nothing while starting (or on an API hiccup); fails if the pod died.
  if ! POD=$(curl -s -H "$H" "$API/pods/$POD_ID" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
if d.get("status") in ("ERROR", "EXITED", "TERMINATED"):
    sys.exit("!!! pod status " + d["status"])
m = [p for p in ((d.get("runtime") or {}).get("ports") or []) if p["private"] == 8000]
print("%s:%s" % (m[0]["ip"], m[0]["public"]) if m else "")'); then exit 1; fi
  [ -n "$POD" ] && break
  sleep 10
done
[ -n "${POD:-}" ] || { echo "!!! the pod never got a public port"; exit 1; }
echo "=== pod at $POD; waiting for vLLM (max 15 min)"
for i in $(seq 90); do curl -sf "http://$POD/health" > /dev/null && break; sleep 10; done
curl -sf "http://$POD/health" > /dev/null || { echo "!!! vLLM never became healthy"; exit 1; }
code=$(curl -s -o /dev/null -w "%{http_code}" "http://$POD/v1/models")
[ "$code" = 401 ] || { echo "!!! the pod answered $code without a key: it is open to the internet"; exit 1; }

echo "=== pointing the cluster's engine Service at the pod"
helm upgrade tf deploy/helm/tokenforge --reset-values --set engine.mode=external --set engine.external.ip="${POD%:*}" \
  --set engine.external.port="${POD#*:}" --set-string engine.apiKey="$VLLM_KEY" --wait --timeout 6m > /dev/null
KEY=$(kubectl exec deploy/tf-gateway -- python -m gateway.keys create phase5 --rpm 1000000 --tpm 1000000000)

for r in 1 2; do
  echo "=== round $r: direct to the pod"
  python bench/loadgen.py --url "http://$POD" --api-key "$VLLM_KEY" --label vllm-int4-kvfp8-pod-direct-r$r-l4 "${LOAD[@]}"
  echo "=== round $r: through the kind cluster"
  python bench/loadgen.py --url http://localhost:8080 --api-key "$KEY" --label k8s-int4-kvfp8-r$r-l4 "${LOAD[@]}"
done
echo "=== benchmarks done"

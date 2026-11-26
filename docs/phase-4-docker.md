# Phase 4: Containers

**Goal:** package the platform as containers: a small, pinned, non-root
gateway image, a compose stack that starts gateway + Redis with one command,
and the official vLLM image as the engine, with weights kept outside every
image. Then scan both images for vulnerabilities and run the whole thing
against a real GPU.

**Summary:**

| | Result |
| --- | --- |
| Gateway image | 282 MB (206 MB Python base + 70 MB virtualenv + 25 KB code), non-root, every dependency pinned |
| Gateway vulnerabilities | 225, all from the Debian base; 0 from our dependencies (pip's 8 removed) |
| vLLM image vulnerabilities | 2,650 (18 critical), 51% from Linux kernel headers the container never runs |
| Containerized stack on a real L4 | 967 tok/s at c=64, the same as Phase 2's in-pod 954 tok/s |
| Gateway cost on real vLLM | not measurable against internet jitter (Phase 3: ~2.5 ms) |

| Piece | Status |
| --- | --- |
| 1. Gateway image: two stages, pinned, non-root | ✅ done |
| 2. Compose stack: gateway + Redis (+ fake engine) | ✅ done |
| 3. Trivy scans | ✅ done |
| 4. The stack against the official vLLM image on a RunPod L4 | ✅ done |

---

## Piece 1: Gateway image (`gateway/Dockerfile`)

```bash
docker build --target gateway -t tokenforge-gateway gateway/
docker build --target fake -t tokenforge-fake-engine gateway/
```

- **Two stages.** The first stage installs dependencies into a virtualenv in
  `/opt/venv`, and only that folder is copied into the runtime stage. All
  dependencies come as prebuilt wheels, so this saves little here, but it
  keeps build tooling out of the runtime image.
- **Every version pinned.** `gateway/requirements.txt` is a full `pip freeze`
  (24 packages), produced inside the base image from four direct pins
  (fastapi 0.162.0, uvicorn[standard] 0.54.0, httpx 0.28.1, redis 8.2.0).
  Pinning only the direct packages would let starlette, pydantic and the
  others drift between builds. The base image is pinned by digest
  (`python:3.14.10-slim-trixie@sha256:22eafb37…`), because the official Python
  image is rebuilt under the same tag.
- **Build context is `gateway/`,** so Docker never uploads the multi-GB
  `venv/` or the results folder.
- **Non-root, numeric UID 10001.** Kubernetes' `runAsNonRoot` (Phase 5) can
  only verify a numeric user.
- **A second build target, `fake`,** adds the fake engine on top of the
  gateway image, so the whole stack runs on a laptop with no GPU. The test-only
  file never ships in the gateway image.
- **No custom vLLM image.** The official `vllm/vllm-openai` image already is
  the engine image. Its entrypoint is `vllm serve`, so a deployment only
  supplies arguments.

## Piece 2: Compose stack (`compose.yaml`)

```bash
docker compose --profile fake up -d --build --wait          # laptop, fake engine
UPSTREAM_URL=http://<ip>:<port> UPSTREAM_API_KEY=<key> \
  docker compose up -d --build --wait                        # real vLLM elsewhere
docker compose exec gateway python -m gateway.keys create alice --rpm 60 --tpm 20000
```

- Ports are bound to `127.0.0.1`, because Docker otherwise publishes them to
  every network the laptop joins. Redis isn't published at all: only the
  gateway reaches it, over compose's internal network.
- Health checks on every service. `--wait` returns only when all are healthy,
  which makes the stack scriptable. The slim image has no `curl`, so the
  checks use Python's `urllib`.
- Redis keeps its data on a named volume, so API keys survive
  `docker compose down`.
- The engine is just `UPSTREAM_URL`. The fake engine sits behind a compose
  profile, and `docker compose down` doesn't stop profiled services, so stop
  them with `docker compose --profile fake down`.
- New in the gateway: `UPSTREAM_API_KEY` is sent to the engine, because a
  RunPod engine is reachable from the internet and must require a key
  (`VLLM_API_KEY`). A test checks that the client's gateway key never reaches
  the engine.

Checked: all services healthy, a key limited to 3 requests/minute gave
`200 200 200 429`, the gateway runs as UID 10001, and keys survived a
down/up.

## Piece 3: Vulnerability scans (Trivy 0.76.2, 2026-11-25)

**Gateway image.** The first scan found exactly what the bare
`python:3.14.10-slim-trixie` image has: 225 Debian findings (0 critical,
54 high) and 8 Python findings. All 8 Python findings were in **pip's bundled
copies** of urllib3, msgpack and setuptools; there was a pip in the base
image and another in our virtualenv. **Our 24 dependencies had none.** The
gateway never runs pip, so I removed it: the virtualenv is created
`--without-pip` and filled with the base image's pip, and the runtime stage
uninstalls the base pip. Python findings went from 8 to 0, and the image from
293 to 282 MB.

Debian findings left (225): 3 have a Debian fix the base image doesn't
include yet (`libzstd1` high, `shadow` ×2), one high is postponed by Debian
(`perl-base`), and the rest have no fix. Most high ones are in `util-linux`,
`login` and `mount`, which the gateway never runs. I didn't add
`apt-get upgrade`, because then the image would depend on the day it's built;
the fix is to update the base digest once Python's image is rebuilt.

**vLLM image** (`vllm/vllm-openai:v0.34.0`, amd64, 10 GB compressed, scanned
on the laptop after the benchmark):

| | Findings | Critical | High | Fixable high/critical |
| --- | --- | --- | --- | --- |
| Ubuntu 24.04 packages (142) | 2,643 | 18 | 333 | 153 |
| of which `linux-libc-dev` | 1,349 | 18 | 317 | 137 |
| Python packages (297) | 7 | 0 | 1 | 1 |

- **51% of all findings and every critical one are in `linux-libc-dev`:** the
  Linux kernel headers, used only to compile code. A container runs on the
  host's kernel, not on these headers, so none of these is reachable at
  runtime. Without it, the vLLM image has 1,294 findings, 0 critical and
  16 high.
- The next largest group is ffmpeg (74 findings in each of its libraries),
  used for audio and video input. A text-only deployment never calls it.
- 297 Python packages produced only 7 findings: `opentelemetry-api` and two
  copies of `setuptools`.
- The vLLM image has 12x the gateway's findings. The size of the image (CUDA,
  a full Ubuntu, ffmpeg, ~300 Python packages) sets the count, not how
  carefully it was built. Fixing them would mean maintaining our own vLLM
  image. Using the official image and updating to new vLLM releases is the
  better trade.

## Piece 4: The stack against real vLLM

```bash
POD=<public-ip>:<external-port> VLLM_KEY=<key> bash bench/phase4.sh
```

- **Pod (RunPod, created with the REST API):** image `vllm/vllm-openai:v0.34.0`,
  1× L4, Secure Cloud, network volume at `/workspace` (weights stay outside
  the image), port 8000 exposed over TCP, `VLLM_API_KEY` set in the
  environment. Start arguments: INT4 weights, FP8 KV cache,
  `--max-model-len 4096`, prefix caching off, the same settings as Phase 2's
  best run.
- **Why this shape:** a RunPod GPU pod is itself a container, so it can't run
  Docker or compose, and the official image has no SSH server. So the pod
  runs only vLLM, and the compose stack (gateway + Redis) runs in Docker
  Desktop on my laptop, reaching the pod over the internet. Exposing the port
  over TCP avoids RunPod's HTTP proxy, which cuts requests off at 100 s.
- **`bench/phase4.sh`** waits for vLLM to load, refuses to continue if
  `/v1/models` answers without a key (an open pod), starts compose pointed at
  the pod, runs loadgen directly at the pod and then through the gateway
  container, then runs Trivy.

Thinking was turned off per request (`chat_template_kwargs`), because the
start command can't safely carry JSON.

| c | Phase 2, inside the pod | Direct, over the internet | Through the gateway container |
| --- | --- | --- | --- |
| 1 | 46.8 tok/s, TTFT 80 ms, TPOT 20.78 | 44.9 tok/s, TTFT 236 ms, TPOT 20.75 | 45.7 tok/s, TTFT 154 ms, TPOT 20.79 |
| 16 | 454.5, 163 ms, 27.63 | 447.7, 324 ms, 27.20 | 451.3, 268 ms, 27.36 |
| 64 | 953.5, 322 ms, 60.28 | 956.5, 679 ms, 57.61 | 967.3, 500 ms, 58.71 |

TTFT and TPOT are p50. Steady state (excluding each level's first wave):

| c=64 | TTFT p50 | TTFT p95 | SLO met |
| --- | --- | --- | --- |
| Phase 2, inside the pod | 231 ms | 855 ms | 190/192 |
| Direct, over the internet | 525 ms | 1,123 ms | 168/192 |
| Through the gateway container | 366 ms | 1,138 ms | 177/192 |

### Findings

**1. Containerized, the stack serves exactly as fast as the hand-run
Phase 2 setup:** 956–967 tok/s at c=64 vs 954, and TPOT at c=1 within
0.05 ms (20.75 / 20.79 vs 20.78). The official image, a pinned tag and weights
on a volume cost nothing in speed.

**2. The internet adds latency to the first token, not to the tokens after
it.** At c=1, TTFT p50 rose from 80 ms (inside the pod) to 154–236 ms, and
p95 from 216 ms to 297–411 ms. TPOT didn't change: once a stream is flowing,
tokens arrive one after another, and the network delay is paid once per
request, not once per token. At c=64, the TTFT tail now breaks the 1 s SLO
for some steady-state requests (168–177/192 met, vs 190/192 inside the pod).
In production the gateway and the GPU sit in the same datacenter. This setup
measures a home connection.

**3. The gateway's cost can't be measured here.** Through the gateway, TTFT
was *lower* than direct (154 vs 236 ms p50 at c=1; the fastest request was
85 vs 124 ms). A gateway can't make requests faster, so this difference comes
from the network, not the gateway: the two runs ran one after the other,
5 minutes apart, over a home internet path whose delay varies by tens of
milliseconds. Phase 3's 2.5 ms gateway cost is far below that variation.
Telling them apart would need interleaved runs (direct, gateway, direct,
gateway…). I didn't do that; the result is "no visible cost," not a
measured number.

**4. Docker Desktop's slowdown didn't appear on real hardware.** The laptop
smoke run had suggested +52 ms TTFT at c=64 through the containerized
gateway. Against real vLLM, throughput through the gateway was 1% higher than
direct and TPOT the same. At ~960 tokens/s the gateway forwards ~960 chunks/s,
well below the ~7,400/s one process handled in Phase 3.

## What this sets up

- **Phase 5 (Kubernetes)** reuses all of this: the gateway image becomes a
  Deployment with `runAsNonRoot` (UID 10001), `/health` is its liveness
  probe, Redis and vLLM become their own workloads, and the official vLLM
  image needs a startup probe long enough for a ~2-minute model load. It
  needs a real Kubernetes node with a GPU, which RunPod can't provide.
- **Not done:** pinning the vLLM image by digest (it's pinned by tag),
  interleaved direct/gateway runs, and a gateway image on a minimal base such
  as distroless.

# TokenForge

A self-hosted, OpenAI-compatible LLM inference platform, built layer by layer
from a deliberately naive baseline to an optimized, autoscaled Kubernetes
deployment. Every layer is benchmarked against the one before it, so each
optimization comes with a measured before/after number.

**Model:** Qwen3-8B  **Hardware:** NVIDIA L4 (24 GB)  **Status:** Phase 0 of 7 complete

---

## Results

All runs use the same fixed workload (`bench/workloads/chat.jsonl`), the same
harness, and the same GPU, so rows are directly comparable.

| Phase | Configuration | Out tok/s @ c=16 | TTFT p95 @ c=16 | TPOT p50 @ c=1 | $ / 1M output tokens |
| --- | --- | --- | --- | --- | --- |
| 0 | Hugging Face `transformers`, FP16, one request at a time | 14.9 | 172,658 ms | 65.6 ms | $13.01 |
| 1 | Quantization (FP8 / INT4) | | | | |
| 2 | vLLM: continuous batching, prefix caching | | | | |
| 6 | Kubernetes, cache-aware routing, autoscaling | | | | |

*c = concurrent requests. TTFT = time to first token. TPOT = time per output token.*

![Latency vs throughput](bench/results/latency_vs_throughput.png)

Better configurations sit further **right** (more throughput) and **lower**
(less latency). The y-axis is logarithmic.

---

## Phase 0 findings: the naive baseline

The baseline serves the model with plain Hugging Face `transformers`, one
request at a time behind a lock. It is slow on purpose: it is the "before"
measurement for everything else.

**1. Concurrency added zero capacity.** Throughput was 14.95 tok/s at
concurrency 1, 4 and 16. Extra users only waited in line.

**2. Queueing fully explains the latency.** Each request took ~10.5 s on
average. At concurrency 4 a request waits behind ~3 others, so predicted TTFT
≈ 31.5 s; measured p50 was 31.3 s. At concurrency 16, p95 TTFT reached 173 s.

**3. Decode is memory-bandwidth-bound.** Generating each token reads all
~16.4 GB of weights. At the L4's ~300 GB/s, the ceiling is ~18 tok/s (55 ms per
token). Measured TPOT was 65.6 ms, an effective ~250 GB/s, about 83% of the
hardware limit. Per-token speed is fine; serving one request at a time is the
problem.

**4. Prefill is ~200x cheaper per token than decode.** TTFT at concurrency 1
grew as ~150 ms + 0.3 ms per prompt token, about 3,500 prompt tokens/s,
versus 15 tok/s for generation. Prefill processes the whole prompt in one
pass over the weights; decode re-reads them for every token.

**5. Goodput collapsed even though throughput didn't move.** Requests meeting
the SLO (TTFT < 1 s, TPOT < 100 ms) fell from 0.095/s to 0.003/s. Only the
first request at each level, which never waited, met it.

**What this sets up:** finding 3 says shrinking the weights should speed up
decode (Phase 1, quantization). Finding 1 says batching requests so one read
of the weights serves many users should multiply throughput (Phase 2, vLLM).

Full write-up: [docs/phase-0-baseline.md](docs/phase-0-baseline.md)

---

## Roadmap

- [x] **Phase 0: Baseline.** Naive HF server and benchmark harness
- [ ] **Phase 1: Model optimization.** Profiling, FP8 and INT4 quantization, quality evals, speculative decoding
- [ ] **Phase 2: Inference engine.** vLLM tuning (continuous batching, prefix caching, chunked prefill), comparison with SGLang
- [ ] **Phase 3: Gateway.** API keys, per-key rate limiting (Redis), streaming proxy
- [ ] **Phase 4: Containers.** Multi-stage Docker images, docker-compose stack
- [ ] **Phase 5: Kubernetes.** Helm chart, GPU scheduling, probes, rolling updates
- [ ] **Phase 6: Routing and autoscaling.** KV-cache-aware routing, KEDA autoscaling on queue depth
- [ ] **Phase 7: Observability and CI.** Grafana dashboards, tracing, benchmark regression gate in CI

---

## Repository layout

```
TokenForge/
├── baseline/
│   └── server.py           # Phase 0: naive HF server (frozen; it's the "before")
├── bench/
│   ├── make_workloads.py   # generates the fixed, seeded workloads
│   ├── loadgen.py          # async load generator: TTFT, TPOT, goodput, $/1M tokens
│   ├── plot.py             # latency vs throughput chart
│   ├── workloads/          # committed once, never edited
│   └── results/            # one JSON per run, with full metadata
├── docs/                   # one write-up per phase
├── requirements-baseline.txt
└── requirements-bench.txt
```

---

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/TokenForge.git
cd TokenForge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-baseline.txt -r requirements-bench.txt
```

**Run the baseline server.** Use a small model on a laptop, or Qwen3-8B on a 24 GB GPU:

```bash
MODEL_ID=Qwen/Qwen3-0.6B uvicorn baseline.server:app --port 8000
```

**Send a request.** The API is OpenAI-compatible, so any OpenAI client works:

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain a KV cache in one sentence."}],"max_tokens":64}'
```

**Benchmark it** (in a second terminal):

```bash
python bench/loadgen.py --workload bench/workloads/chat.jsonl \
  --concurrency 1 4 16 --num-requests 32 --ignore-eos --max-tokens-cap 256 \
  --gpu-cost-per-hour 0.70 --label my-run
python bench/plot.py bench/results/*.json
```

---

## Benchmark methodology

**Workloads.** Three fixed JSONL files generated once from a fixed seed
(`make_workloads.py --seed 42`) and committed:

| Workload | Shape | Stresses |
| --- | --- | --- |
| `chat` | Log-normal input and output lengths (median ~150 words in, ~200 tokens out) | General serving |
| `shared_prefix` | Same ~2k-token system prompt + short question | Prefix caching, cache-aware routing |
| `long_context` | ~8k-token input, 128-token output | Prefill, KV cache memory |

**Load.** Closed-loop: *c* virtual users, each sending its next request as
soon as the previous one finishes. Each concurrency level sends the same 32
requests (shuffled with a fixed seed) after 2 warmup requests.

**Settings for all reported runs.** `temperature=0`, `ignore_eos=true`
(every request generates exactly its `max_tokens`), `max_tokens` capped at
256, streaming with server-reported token counts (`stream_options.include_usage`).

**Metrics.**

| Metric | Definition |
| --- | --- |
| TTFT | Request sent → first content token received (queueing + prefill) |
| TPOT | (end-to-end − TTFT) ÷ (output tokens − 1) (decode speed) |
| Output throughput | Total output tokens ÷ wall-clock time for the level |
| Goodput | Requests per second meeting the SLO: TTFT < 1,000 ms and TPOT < 100 ms |
| $ / 1M output tokens | GPU $/hour ÷ output tokens per hour × 1,000,000 |

Latencies are reported as percentiles (p50, p95, p99), not averages.

**Environment for reported results.** NVIDIA L4 (23 GB usable), driver
595.91.07, RunPod at $0.70/hour, Python 3.12, server and load generator on
the same machine (no network latency included). Every result file in
`bench/results/` records its full configuration, git commit and hardware.
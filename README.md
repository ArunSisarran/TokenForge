# TokenForge

A self-hosted, OpenAI-compatible LLM inference platform, built layer by layer
from a deliberately naive baseline to an optimized, autoscaled Kubernetes
deployment. Every layer is benchmarked against the one before it, so each
optimization comes with a measured before/after number.

**Model:** Qwen3-8B  **Hardware:** NVIDIA L4 (24 GB)  **Status:** Phases 0–2 complete, Phase 3 next

---

## Results

All runs use the same fixed workload (`bench/workloads/chat.jsonl`), the same
harness, and the same GPU, so rows are directly comparable.

| Phase | Configuration | Out tok/s @ c=16 | TTFT p95 @ c=16 | TPOT p50 @ c=1 | $ / 1M output tokens @ c=16 |
| --- | --- | --- | --- | --- | --- |
| 0 | Hugging Face `transformers`, FP16, one request at a time | 14.9 | 172,658 ms | 65.6 ms | $13.01 |
| 1 | vLLM, FP16, one request at a time | 16.5 | 163,507 ms | 59.9 ms | $11.77 |
| 1 | vLLM, FP8 weights (W8A8), one request at a time | 26.3 | 102,331 ms | 37.6 ms | $7.39 |
| 1 | vLLM, INT4 weights (GPTQ W4A16), one request at a time | 46.7 | 60,404 ms | 20.8 ms | $4.17 |
| 1 | + EAGLE-3 speculative decoding (k=3) | 74.8 | 35,080 ms | 13.2 ms | $2.60 |
| 2 | vLLM INT4, continuous batching | 432.6 | 1,395 ms | 20.8 ms | $0.45 |
| 2 | + FP8 KV cache | 454.5 | 1,504 ms | 20.8 ms | $0.43 |
| 2 | INT4 + EAGLE-3 (k=3), batching | 541.0 | 1,585 ms | 13.1 ms | $0.36 |
| 2 | SGLang 0.5.23, INT4, batching | 420.9 | 1,749 ms | 20.2 ms | $0.46 |
| 6 | Kubernetes, cache-aware routing, autoscaling | | | | |

*c = concurrent requests. TTFT = time to first token. TPOT = time per output token.*

Phase 0–1 rows serve one request at a time, so their cost is the same at
every concurrency. At c=64, vLLM INT4 + FP8 KV cache reaches **954 tok/s at
$0.20 per 1M tokens** (64x the baseline), while EAGLE-3 falls behind plain
decoding (780 vs 839 tok/s).

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

**3. Decode is memory-bandwidth-bound.** Generating each token reads every
transformer-block weight plus the `lm_head`, about 15.2 GB (the 1.2 GB
embedding table is a single-row lookup, not a full read). At the L4's
~300 GB/s, the ceiling is ~51 ms per token. Measured TPOT was 65.6 ms, about
77% of the hardware limit. Per-token speed is decent; serving one request at
a time is the problem.

**4. Prefill is ~200x cheaper per token than decode.** TTFT at concurrency 1
grew as ~160 ms + 0.31 ms per prompt token, about 3,200 prompt tokens/s,
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

## Phase 1 findings: model-level optimization

Before optimizing, `optimize/model_math.py` predicts memory use and decode
ceilings from the model config, and `optimize/profile_request.py` runs
prefill and decode by hand under the PyTorch profiler to see where GPU time
actually goes.

**Model math predictions (Qwen3-8B on L4, 512-token context):**

| Weights | Weight memory | KV cache room | Min time per token | Max tok/s |
| --- | --- | --- | --- | --- |
| FP16 | 16.4 GB | ~36k tokens | 50.7 ms | ~20 |
| FP8 | 9.4 GB | ~84k tokens | 27.6 ms | ~36 |
| INT4 | 6.0 GB | ~107k tokens | 16.0 ms | ~63 |

KV cache: 144 KB per token (36 layers × 8 KV heads × 128 dims × K and V × 2 bytes).
Grouped-query attention (8 KV heads for 32 query heads) makes it 4x smaller.

**Where one decode step's time goes** (HF FP16, from the profiler trace):

| | 512-token prompt | 2048-token prompt |
| --- | --- | --- |
| Total step time | 67.5 ms | 68.3 ms |
| Matrix multiplications | 58.8 ms (87%) | 59.0 ms (86%) |
| Other small kernels | 4.6 ms (7%) | 6.3 ms (9%) |
| GPU idle (gaps) | 3.4 ms (5%) | 3.0 ms (4%) |
| Kernels launched per step | 2,074 | 2,074 |
| Model-math ceiling | 50.7 ms | 55.0 ms |

**1. Kernel launch gaps are small:** only ~5% of the step is GPU idle time, so CUDA graphs can recover at most that.

**2. The matmuls already run near the practical limit:** ~257 GB/s, 86% of the L4's 300 GB/s spec.

**3. Decode is memory-bound:** the 253 `gemv` kernels split their time 79/21 between MLP + `lm_head` and attention, matching their 80/20 split in bytes.

**4. Hugging Face copies the whole KV cache every step:** 72 `torch.cat` kernels per step, rising from 2.2% to 4.1% of the step from 512 to 2048 tokens of context. PagedAttention avoids this.

**5. Prefill is compute-bound but reaches only about a third of peak:** ~40–43 TFLOPS against ~120, with unfused elementwise ops at 9–17% of prefill time.

**What this means for the rest of Phase 1:** at one request at a time, the HF
baseline is already near the practical bandwidth limit, so a better engine
alone should give only a modest per-token gain. The large levers are reading
fewer bytes (quantization) and serving many requests per read of the
weights (batching, Phase 2).

**Speed and quality per precision** (vLLM 0.34, one request at a time, chat workload):

| | TPOT p50 | Speed-up vs HF | $ / 1M tokens | GSM8K (5-shot) | KV cache room |
| --- | --- | --- | --- | --- | --- |
| FP16 | 59.9 ms | 1.1x | $11.77 | 87.64 | 10,240 tokens |
| FP8 | 37.6 ms | 1.7x | $7.39 | 86.73 | 56,320 tokens |
| INT4 | 20.8 ms | 3.2x | $4.17 | 85.75 | 79,360 tokens |

**6. The engine alone gave +10.6% throughput** (TPOT 65.6 → 59.9 ms), inside my 10–15% prediction: vLLM's whole step now takes about as long as HF's matmuls alone.

**7. Quantization sped up decode almost in proportion to bytes:** INT4 reads 3.2x fewer bytes than FP16 and ran 2.9x faster; FP8 reads 1.8x fewer and ran 1.6x faster.

**8. I was wrong that INT4 wouldn't help prefill:** the chat prompts (~295 tokens) are below the L4's ~400-token compute-bound point, so TTFT p50 fell 110 → 80 ms for INT4 (and 69 ms for FP8, which also gets 2x tensor-core FLOPs).

**9. Cost fell 3.1x for 1.9 GSM8K points** ($13.01 → $4.17 per 1M tokens with INT4). Goodput at concurrency 16 is still 1 in 32, so batching (Phase 2) is the next big lever.

**10. An FP8 KV cache made long-context decode 7.2% faster** (24.55 → 22.78 ms at ~6.8k tokens, matching the 0.5 GB of KV bytes it saves) for +0.10% perplexity, and KV room went 86k → 166k tokens. vLLM also switched the attention backend for FP8 KV (FlashAttention → FlashInfer); a control run showed the backend accounts for only 0.10 ms of the 1.77 ms. On short chat prompts it changed nothing, as predicted.

**11. EAGLE-3 speculative decoding made INT4 decode 1.6x faster** (mean TPOT 20.9 → 12.7 ms, $4.17 → $2.60 per 1M tokens) with 2.35 tokens accepted per target pass. Checking 4 tokens instead of 1 cost only ~0.2 ms, because decode is memory-bound; drafting took 2.9 ms per token, close to the 2.7 ms predicted from its bytes.

Full write-up: [docs/phase-1-optimization.md](docs/phase-1-optimization.md)

---

## Phase 2 findings: engine tuning under load

Phase 1 served one request at a time. Phase 2 lets vLLM batch requests,
sweeps concurrency from 1 to 64, and changes one engine flag per run.

![Phase 2 batching](bench/results/phase2-batching.png)

**1. Batching gave 18x throughput for 3.3x TPOT.** INT4 went from 46.7 to 839 tok/s between c=1 and c=64 (TPOT 20.8 → 69.6 ms): $0.23 per 1M tokens, 56x cheaper than the baseline. One read of the weights now serves up to 64 sequences.

**2. At c=64 the GPU becomes compute-bound.** It does ~35 TFLOP/s of useful math (4.64 requests/s × 462 tokens × 16.4 GFLOP), close to the 40–43 TFLOPS the best Phase 1 prefill reached. That's why INT4's lead over FP8 shrinks from 1.76x at c=1 to 1.11x at c=64: fewer weight bytes stop helping once math is the limit.

**3. FP16 runs out of KV cache:** ~15.9k tokens of room fits only ~30 sequences, so at c=64 throughput stops at 273 tok/s and TTFT p50 reaches 23.6 s.

**4. `max-num-seqs` trades TTFT for TPOT, and queueing math predicts it.** Capped at 8 with 64 users, TPOT stays at 25 ms, but each request waits for 7 others ahead of it (~4.3 s each): ~30 s predicted, 32.2 s measured. With a 100 ms TPOT SLO, no cap is best.

**5. `max-num-batched-tokens` only moves latency between TTFT and TPOT.** On 6.7k-token prompts, budgets of 512 / 2048 / 8192 gave the same throughput (±3%) and E2E (23.1–23.8 s): the total prefill work doesn't change. Small chunks even out TPOT across requests; big ones give faster decodes between long stalls.

**6. Prefix caching gave 4.6x throughput on a shared ~1.7k-token system prompt** (175 → 800 tok/s at c=32, SLO 0% → 100%), with a 96.5% token hit rate. TTFT at c=1 fell 510 → 63 ms. TPOT under load improved 4.4x too, because prefill no longer competes with decode.

**7. The FP8 KV cache helps only under load:** +0% at c=1, +13.6% at c=64 (TPOT 69.6 → 60.3 ms), against ~6.6 ms predicted from bytes. FlashInfer also changed (as in Phase 1), and I didn't run that control at batch 64.

**8. EAGLE-3 wins at low load and loses at high load:** 1.61x at c=1, 1.05x at c=32, 0.93x at c=64, with acceptance unchanged (L = 2.42). Checking 4 tokens to keep 2.42 costs 1.65x the compute per token, which is free when memory-bound and expensive when compute-bound.

**9. SGLang 0.5.23 matched vLLM within ±3%** on throughput and TPOT, but had a worse steady-state TTFT p95 at c=64 (1.32 s vs 0.73 s) and took ~10 minutes to start (480 s of CUDA graph capture) vs ~2 for vLLM.

**10. Most SLO misses at c=64 came from the benchmark's start-up burst.** Closed-loop load starts each level with all c users at once. For INT4 at c=64, all 57 requests over 1 s TTFT were in that first wave; after it, 189/192 met the SLO.

Full write-up: [docs/phase-2-engine.md](docs/phase-2-engine.md)

---

## Roadmap

- [x] **Phase 0: Baseline.** Naive HF server and benchmark harness
- [x] **Phase 1: Model optimization.**
    - [x] Model math: memory, KV cache, and decode ceilings per precision
    - [x] Profiling: prefill vs decode, kernel breakdown
    - [x] FP8 and INT4 quantization
    - [x] Quality evals (perplexity, task accuracy)
    - [x] Speed benchmarks: FP16 vs FP8 vs INT4
    - [x] FP8 KV cache
    - [x] Speculative decoding
- [x] **Phase 2: Inference engine.**
    - [x] Continuous batching: FP16 vs FP8 vs INT4, c=1 to 64
    - [x] Scheduler limits: `max-num-seqs`, `max-num-batched-tokens`
    - [x] Prefix caching
    - [x] FP8 KV cache and speculative decoding under load
    - [x] SGLang comparison
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
│   ├── phase2.sh           # Phase 2 engine experiments, run back to back on the GPU
│   ├── workloads/          # committed once, never edited
│   └── results/            # one JSON per run, with full metadata
├── optimize/
│   ├── model_math.py       # predicts memory, KV cache size, decode ceilings
│   └── profile_request.py  # profiles prefill and decode, exports a GPU trace
├── docs/                   # one write-up per phase
├── requirements.txt        # all Python dependencies except quantization
└── requirements-quantize.txt  # llmcompressor, in its own venv (conflicts with vLLM)
```

---

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/TokenForge.git
cd TokenForge
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
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
requests (shuffled with a fixed seed) after 2 warmup requests. From Phase 2
on, `--min-rounds 4` sends at least 4 requests per user, so high-concurrency
levels reach steady state. All users start together, so the first wave of
each level is a burst; Phase 2 reports SLO attainment with and without it.

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
595.91.07 (Phase 0) and 595.112.07 (Phase 1), RunPod at $0.70/hour, Python
3.12, server and load generator on the same machine (no network latency
included). vLLM runs use vLLM 0.34.0 with prefix caching off (except the
Phase 2 prefix caching runs) and Qwen3's thinking mode off, to match the
baseline. SGLang runs use SGLang 0.5.23 with its radix cache off. Every result file in
`bench/results/` records its full configuration, git commit and hardware.
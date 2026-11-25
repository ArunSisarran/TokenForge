# Phase 1: Model-level optimization

**Goal:** make each token cheaper at the model level (quantization, KV cache,
speculative decoding) and be able to explain *why* each change helps.

**Approach:** predict from first principles before measuring, then change
one variable at a time.

**Summary** (Qwen3-8B on one L4, chat workload, one request at a time):

| Step | TPOT | Out tok/s | $ / 1M tokens | Quality cost |
| --- | --- | --- | --- | --- |
| HF FP16 baseline (Phase 0) | 65.6 ms | 14.95 | $13.01 | |
| vLLM FP16 (engine only) | 59.9 ms | 16.53 | $11.77 | |
| vLLM FP8 weights | 37.6 ms | 26.31 | $7.39 | GSM8K −0.9 pts (noise), PPL +1.0% |
| vLLM INT4 weights | 20.8 ms | 46.64 | $4.17 | GSM8K −1.9 pts, PPL +3.9% |
| INT4 + EAGLE-3 speculative decoding | 13.2 ms | 74.80 | $2.60 | none expected (greedy) |

FP8 KV cache: no change on chat, −6.8% TPOT at 6.8k-token context, and 1.93x
KV room.

Overall: 5.0x throughput and 5.0x lower cost per token than the baseline,
for about 2 GSM8K points. Every step except the engine swap removed bytes
read per token, and each one landed close to its byte-count prediction.
Concurrency still adds nothing, because every run here served one request at a
time. That's Phase 2.

| Piece | Status |
| --- | --- |
| 1. Model math: memory, KV cache and decode ceilings per precision | ✅ done |
| 2. Profiling: where prefill and decode time goes | ✅ done |
| 3. FP8 and INT4 quantization | ✅ done |
| 4. Quality evals (perplexity, task accuracy) | ✅ done |
| 5. Speed benchmarks: FP16 vs FP8 vs INT4 in vLLM | ✅ done |
| 6a. FP8 KV cache | ✅ done |
| 6b. Speculative decoding | ✅ done |

---

## Piece 1: Model math (`optimize/model_math.py`)

Before running anything on a GPU, I wanted numbers to check measurements
against. The script reads only the model's `config.json` (no GPU and no
weight download) and computes, for each weight precision:

- GPU memory used by the weights
- bytes read from memory to generate one token at batch size 1
- the fastest possible decode speed: bytes per token ÷ memory bandwidth
- how many tokens of KV cache fit in the memory that's left

```bash
python optimize/model_math.py --model Qwen/Qwen3-8B
```

### Counting parameters

Each transformer layer has an attention block (Q, K, V and output
projections) and an MLP (gate, up and down projections). Qwen3-8B:

| | Value |
| --- | --- |
| Layers | 36 |
| Query heads / KV heads | 32 / 8 (grouped-query attention) |
| Head dim | 128 |
| Parameters | 8.19B total |
| Transformer blocks | 6.95B |
| Embedding table | 0.62B |
| `lm_head` | 0.62B (not tied to the embeddings, so it counts separately) |

### KV cache per token

Every layer stores one key vector and one value vector per KV head, for each
token:

```
2 (K and V) × 36 layers × 8 KV heads × 128 dims × 2 bytes = 147,456 bytes ≈ 144 KB
```

Grouped-query attention matters a lot here. With 32 KV heads (one per query
head) the cache would be 4x larger, 576 KB per token.

### Predictions per precision

Assumptions: only the transformer blocks are quantized, while the embedding
table and `lm_head` stay 16-bit (what typical recipes do). The L4 has 300 GB/s
of bandwidth and 23,034 MiB of memory, of which the server may use 90%.

| Weights | Weight memory | KV cache room | Bytes read per token @ 512 ctx | Min TPOT | Max tok/s |
| --- | --- | --- | --- | --- | --- |
| FP16 | 16.4 GB | ~36k tokens | 15.2 GB | 50.7 ms | ~20 |
| FP8 | 9.4 GB | ~84k tokens | 8.3 GB | 27.6 ms | ~36 |
| INT4 | 6.0 GB | ~107k tokens | 4.8 GB | 16.0 ms | ~63 |

Bytes read per decode step = block weights + `lm_head` + context × KV per
token. The embedding table isn't counted: a decode step looks up one row of
it, not the whole table.

### What the math says

- **Quantization should cut decode time almost in proportion to bytes.** FP8
  roughly halves the bytes read per token, and INT4 cuts them to about a third.
- **The 16-bit `lm_head` becomes a big share at INT4.** It's 1.2 GB of the
  4.8 GB read per token, about 26%. That makes it the next target once the
  blocks are 4-bit.
- **The KV cache grows with context.** At 8k tokens of context, reading the KV
  cache adds 1.2 GB per step, about 8% on top of FP16 weights and 25% on top
  of INT4.
- **Smaller weights leave room for more concurrent sequences.** Freeing memory
  from weights roughly triples KV cache room from FP16 to INT4. This only pays
  off with an engine that batches requests (Phase 2).

---

## Piece 2: Profiling one request (`optimize/profile_request.py`)

The script runs prefill and decode by hand, the same steps `generate()` runs
internally, so each phase can be timed and traced separately:

- **Prefill:** one forward pass over the whole prompt with
  `logits_to_keep=1`, so only the last position's logits are computed.
- **Decode:** one forward pass per new token, reusing the KV cache.
- **Timing:** `torch.cuda.synchronize()` before and after each call. GPU
  work is asynchronous, so without it I'd be timing how fast Python queues
  kernels, not how fast the GPU runs them.
- **Warmup:** one prefill and 5 decode steps are thrown away first, to skip
  one-time costs like kernel selection and memory allocation.
- **Trace:** `torch.profiler` records one prefill and 8 decode steps to a
  Chrome trace, which I read at [ui.perfetto.dev](https://ui.perfetto.dev).

```bash
python -m optimize.profile_request --model Qwen/Qwen3-8B --prompt-tokens 512
python -m optimize.profile_request --model Qwen/Qwen3-8B --prompt-tokens 2048 \
  --trace optimize/traces/trace-2048.json
```

### Where one decode step's time goes (HF FP16)

| | 512-token prompt | 2048-token prompt |
| --- | --- | --- |
| Total step time | 67.5 ms | 68.3 ms |
| Matrix multiplications | 58.8 ms (87%) | 59.0 ms (86%) |
| Other small kernels | 4.6 ms (7%) | 6.3 ms (9%) |
| GPU idle (gaps between kernels) | 3.4 ms (5%) | 3.0 ms (4%) |
| Kernels launched per step | 2,074 | 2,074 |
| Model-math ceiling | 50.7 ms | 55.0 ms |

At 512 tokens the measured step runs at 75% of the model-math ceiling
(50.7 ÷ 67.5), consistent with the 65.6 ms TPOT from Phase 0.

### Findings

**1. My prediction about launch overhead was wrong.** With ~2,000 kernels per
step, I expected the gaps between kernel launches to explain most of the
distance from the ceiling. The trace showed only ~5% idle time. Each matmul
runs about 230 µs, which is long enough for Python to queue the next kernels,
so the GPU rarely waits. CUDA graphs, which remove launch overhead, can
recover at most that 5% here.

**2. The matmul kernels already run near the practical limit.** They read
~15.1 GB in 58.8 ms, about 257 GB/s, or 86% of the L4's 300 GB/s spec. Real
kernels rarely get past 85–90% of spec bandwidth, so there's little left to
gain in the matmuls themselves at FP16.

**3. Kernel timing confirms decode is memory-bound.** Decode runs 253 `gemv`
(matrix × vector) kernels per step: 7 per layer (Q, K, V, O, gate, up, down)
× 36 layers, plus one for the `lm_head`. Their time splits 79% / 21% between
the MLP + `lm_head` and the attention projections. Their bytes split 80% /
20%. When time tracks bytes this closely, the kernels are limited by memory
bandwidth, not compute.

**4. Hugging Face copies the whole KV cache every step.** Each step runs 72
`torch.cat` copy kernels (2 per layer) that build a new, longer cache tensor
to append one token. Their share of the step roughly doubled from 2.2% at 512
tokens of context to 4.1% at 2048. Because each copy is the size of the whole
cache, the total cost grows quadratically with generation length. vLLM's
PagedAttention avoids this by preallocating fixed-size cache blocks and
writing new tokens in place.

**5. Prefill is compute-bound but uses about a third of peak.** The GPU was
97–99% busy during prefill.

| Prompt | Time | Prompt tok/s | Achieved |
| --- | --- | --- | --- |
| 512 tokens | 180 ms | ~2,800 | ~40 TFLOPS |
| 2048 tokens | 659 ms | ~3,100 | ~43 TFLOPS |

That's against roughly 120 TFLOPS FP16 peak. Unfused elementwise ops (for
example, RMSNorm's separate squaring step) took 9% of prefill at 512 tokens
and 17% at 2048. Kernel fusion in engines like vLLM removes most of that.
Attention itself was under 2% of decode time at these context lengths.

### What this means for the rest of Phase 1

At batch size 1, the HF baseline is already close to the bandwidth limit. A
better engine on its own should give only a modest per-token gain: I predict
about 10–15% for vLLM FP16 at `--max-num-seqs 1`. The large levers are:

1. **Reading fewer bytes per token:** quantization (piece 3).
2. **Serving many requests per read of the weights:** batching (Phase 2).

---

## Piece 3: Quantization (`optimize/quantize.py`)

**Plan:** `optimize/quantize.py --scheme {fp8,int4}` using LLM Compressor.

- **FP8** (`FP8_DYNAMIC`, W8A8): 8-bit weights and activations, with
  activation scales computed at runtime, so no calibration data is needed.
- **INT4** (GPTQ, `W4A16`, group size 128): 4-bit weights, 16-bit
  activations. GPTQ uses 512 calibration samples from
  `HuggingFaceH4/ultrachat_200k` to choose roundings that minimize each
  layer's output error.
- `lm_head` stays 16-bit in both, because it's sensitive to quantization error.

**Predictions to check:**

| | On-disk size | Expected effect |
| --- | --- | --- |
| BF16 | ~16.4 GB | Baseline |
| FP8 | ~9.4 GB | Faster prefill (FP8 tensor cores) and faster decode (half the bytes) |
| INT4 | ~6.1 GB (including ~3% for group-128 scales) | Fastest decode, little or no prefill gain (math still runs in 16-bit) |

```bash
python -m optimize.quantize --scheme fp8
python -m optimize.quantize --scheme int4
```

### Results (L4, llmcompressor 0.15)

| | Predicted size | Measured size | Time to quantize | Peak CPU RAM |
| --- | --- | --- | --- | --- |
| FP8 | 9.43 GB | 9.44 GB | 0.1 min | 16.3 GB |
| INT4 | 6.07 GB | 6.07 GB | 27.9 min | not recorded |

### Findings

**1. Sizes on disk match the model math almost exactly.** FP8 came out at
6.95B block weights × 1 byte + 2.48 GB of 16-bit embeddings and `lm_head`.
INT4 came out at 6.95B × 0.5 bytes + 0.11 GB of group-128 scales + the same
2.48 GB. The 16-bit embeddings and `lm_head` are now 41% of the INT4 file.

**2. GPTQ took hundreds of times longer than FP8.** FP8_DYNAMIC only computes one scale
per output row from the weights and rounds, so it needs no data and took
a few seconds (the script rounds to 0.1 min). GPTQ ran 512 calibration conversations through every layer
and solved a rounding problem per weight matrix, which took 28 minutes. That
cost is paid once, offline.

**3. Peak CPU RAM for FP8 (16.3 GB) was the BF16 model itself.** The model
was loaded on the CPU in BF16 (16.4 GB) and quantized in place, so nothing
else meaningful was added.

**4. Both models still produce fluent answers, which proves little.** Both
read "KV cache" as a generic key-value store rather than the transformer
cache. That comes from the ambiguous prompt, not from quantization damage,
and a single greedy sample can't measure quality. Piece 4 measures that with
perplexity and task accuracy against BF16.

---

## Piece 4: Quality evals (lm-evaluation-harness)

Same evals, same settings, on all three models: WikiText word perplexity
(62 documents) and GSM8K (all 1,319 problems, 5-shot, greedy). The vLLM
backend runs the real quantized kernels: `CutlassFP8ScaledMMLinearKernel`
for FP8 and `MarlinLinearKernel` for INT4. No chat template, so absolute
GSM8K scores won't match Qwen's published numbers; the differences between
the three models are what's being measured.

```bash
lm_eval --model vllm \
  --model_args pretrained=$MODEL,dtype=auto,max_model_len=4096,gpu_memory_utilization=0.85 \
  --tasks wikitext,gsm8k --batch_size auto --output_path optimize/evals/
```

Environment: NVIDIA L4 (driver 595.112.07), vLLM 0.34.0, torch 2.14.0
(CUDA 13.0 build), transformers 5.22.2, lm-eval 0.4.13. Raw results are in
`optimize/evals/*/results_*.json`.

### Results

| | WikiText word PPL ↓ | Δ PPL | GSM8K strict ↑ | Δ GSM8K | GSM8K recovery |
| --- | --- | --- | --- | --- | --- |
| BF16 | 12.743 | | 87.64 ± 0.91 | | |
| FP8 (W8A8 dynamic) | 12.867 | +0.97% | 86.73 ± 0.93 | −0.91 pts | 99.0% |
| INT4 (GPTQ W4A16 g128) | 13.242 | +3.92% | 85.75 ± 0.96 | −1.90 pts | 97.8% |

GSM8K flexible-extract shows the same pattern: 88.32 → 87.34 → 86.50.
Recovery = quantized score ÷ BF16 score.

What vLLM reported while loading each model (85% of 22.04 GiB, 4,096-token
max length):

| | Weights in GPU memory | KV cache | Max concurrent 4k-token requests |
| --- | --- | --- | --- |
| BF16 | 15.27 GiB | 10,240 tokens | 2.5 |
| FP8 | 8.80 GiB | 56,320 tokens | 13.75 |
| INT4 | 5.68 GiB | 79,360 tokens | 19.4 |

### Findings

**1. FP8 is nearly free in quality.** Perplexity rose 1% and GSM8K fell
0.9 points, which is within one standard error. On these evals I can't tell
FP8 apart from BF16.

**2. INT4 costs about 4x more quality than FP8, but still keeps ~98% of
it.** Perplexity rose 3.9% and GSM8K fell 1.9 points. Treating the two
scores as independent, the GSM8K gap is about 1.4 standard errors (√(0.91² +
0.96²) ≈ 1.3), so it's suggestive but not proven at 95%. Both models answered
the same 1,319 questions, so a paired comparison would be much tighter; that
needs `--log_samples`, which this run didn't save. Perplexity has no sampling
noise here (fixed text, deterministic scoring), so its ordering is real.

**3. Perplexity and GSM8K agree on the ordering.** BF16 > FP8 > INT4 on
both. The fewer bits per weight, the more the output distribution drifts,
and GSM8K's long multi-step generations turn that drift into wrong answers.

**4. Weight memory matched the model math.** 15.27 / 8.80 / 5.68 GiB versus
predicted 16.4 / 9.43 / 6.07 GB (15.3 / 8.8 / 5.65 GiB).

**5. KV cache room was far smaller than the model math predicted, and that
hits BF16 hardest.** I predicted ~36k / 84k / 107k tokens at 90% memory.
vLLM got 10k / 56k / 79k at 85%. The model math ignored ~1.3 GiB of peak
activations and other runtime memory, and 5% less utilization costs another
~1.1 GiB. That overhead is roughly fixed, so it takes most of BF16's small
leftover: BF16 can hold only 2.5 full-length requests at once, while INT4
holds 19. **Quantization's biggest win for serving may be batch size, not
per-token speed.** That's what Phase 2 tests.

**6. Eval wall time hints at that already.** BF16 took 38 min, FP8 17 min,
INT4 18 min. These weren't controlled benchmarks (they include engine
start-up), but BF16's tiny KV cache limits how many sequences vLLM can batch.
INT4 was no faster than FP8 despite more KV room and fewer bytes: evals are
mostly prefill, which is compute-bound, and FP8 runs on FP8 tensor cores
while W4A16 still does FP16 math. Piece 5 measures this properly.

---

## Piece 5: Speed benchmarks in vLLM

Two comparisons, one variable each:

| Comparison | What changes | What it isolates |
| --- | --- | --- |
| HF FP16 (Phase 0) → vLLM FP16 | the engine | fused kernels, CUDA graphs, paged KV cache |
| vLLM FP16 → vLLM FP8 / INT4 | the weights | the effect of reading fewer bytes per token |

All three runs use `--max-num-seqs 1`, so vLLM serves one request at a time
like the baseline; batching is Phase 2. Prefix caching is off, because every
concurrency level sends the same 32 requests and cache hits would make TTFT
look better than it is. Thinking is off, matching the baseline's chat
template. The FP16 run uses `--dtype float16` to match the baseline exactly.

```bash
vllm serve $MODEL --dtype {float16|auto} --max-num-seqs 1 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}'
python bench/loadgen.py --workload bench/workloads/chat.jsonl --model $MODEL \
  --concurrency 1 4 16 --num-requests 32 --ignore-eos --max-tokens-cap 256 \
  --gpu-cost-per-hour 0.70 --label vllm-{fp16,fp8,int4}-seq1-l4
```

Environment: as in piece 4 (vLLM 0.34.0, driver 595.112.07). Results are in
`bench/results/vllm-*-seq1-l4-chat-*.json`.

### Results (concurrency 1)

| | Out tok/s | TPOT p50 | Model-math ceiling | % of ceiling | Effective GB/s | TTFT p50 | TTFT p95 | $ / 1M tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HF FP16 (Phase 0) | 14.95 | 65.6 ms | 50.7 ms | 77% | 232 | 225 ms | 419 ms | $13.01 |
| vLLM FP16 | 16.53 | 59.9 ms | 50.7 ms | 85% | 254 | 110 ms | 234 ms | $11.77 |
| vLLM FP8 | 26.31 | 37.6 ms | 27.6 ms | 73% | 221 | 69 ms | 153 ms | $7.39 |
| vLLM INT4 | 46.64 | 20.8 ms | 16.0 ms | 77% | 231 | 80 ms | 214 ms | $4.17 |

Effective GB/s = model-math bytes per token (512-token context) ÷ TPOT.
Throughput and TPOT were the same at concurrency 4 and 16 (within 0.1%), as
expected with one sequence at a time; only TTFT grew, from queueing.

### Speed and quality together

| | TPOT p50 | Speed-up vs HF | $ / 1M tokens | GSM8K strict | KV cache room |
| --- | --- | --- | --- | --- | --- |
| FP16 / BF16 | 59.9 ms | 1.1x | $11.77 | 87.64 | 10,240 tokens |
| FP8 | 37.6 ms | 1.7x | $7.39 | 86.73 | 56,320 tokens |
| INT4 | 20.8 ms | 3.2x | $4.17 | 85.75 | 79,360 tokens |

Quality and KV room are from piece 4, where the reference model ran in BF16.
FP16 and BF16 move the same bytes, so the speed is the same.

### Findings

**1. The engine alone gave 10.6% more throughput, inside my 10–15%
prediction.** TPOT fell from 65.6 to 59.9 ms. vLLM's whole decode step is
now about as long as just the matmuls were in the HF trace (58.8 ms in piece
2). That fits vLLM removing almost all of the non-matmul time: the idle gaps
(CUDA graphs), the small unfused kernels, and the `torch.cat` cache copies.
I didn't profile vLLM, so that breakdown is inferred, not measured. What the
engine can't change is the bytes: at 254 GB/s it's at the practical limit,
so FP16 decode has nothing left to give.

**2. Quantization sped up decode almost in proportion to bytes, but not
quite.** INT4 reads 3.2x fewer bytes than FP16 and ran 2.9x faster. FP8 reads
1.8x fewer and ran 1.6x faster. Treating FP16 and INT4 as two points on a
line (time = bytes ÷ bandwidth + fixed cost) gives 266 GB/s plus ~2.8 ms of
fixed cost per step. The fixed cost (norms, attention, sampling) doesn't
shrink with the weights, so it's a bigger share of a short step.

**3. FP8 is ~3.6 ms per step slower than that line predicts.** At 266 GB/s +
2.8 ms, FP8 should take ~34 ms; it took 37.6. My hypothesis: FP8_DYNAMIC
quantizes the activations at runtime, which adds a small kernel before each
of the 252 linear layers per step. At batch size 1 that is pure overhead.
This is a hypothesis to confirm with a vLLM trace, not a measured result.

**4. My prediction that INT4 wouldn't speed up prefill was wrong for this
workload.** TTFT p50 went 110 → 69 ms for FP8 and 110 → 80 ms for INT4. The
chat prompts average only 295 tokens. On the L4 a matmul becomes
compute-bound at about 400 tokens (121 TFLOPS ÷ 300 GB/s ≈ 400 FLOPs per
byte, and FP16 weights give one FLOP per byte per token), so these prefills
are still partly limited by reading the weights, and reading 4-bit weights
helps. A rough floor for a 295-token prefill, max(bytes ÷ bandwidth,
FLOPs ÷ peak), is 51 / 27 / 34 ms for FP16 / FP8 / INT4. That ranks the
three in the same order I measured: FP8 is fastest because it is the only one that
cuts both bytes and FLOPs time (FP8 tensor cores at ~242 TFLOPS). Each
measured TTFT is ~40–60 ms above its floor (HTTP, tokenization, scheduling).
For long prompts (the `long_context` workload) I'd expect INT4's prefill
advantage to disappear and FP8 to pull further ahead.

**5. Some of the HF → vLLM TTFT gain is a measurement artifact.** The HF
streamer sends whole words (3,857 chunks for 5,028 tokens), so HF's TTFT
includes a little word buffering. vLLM sends about one chunk per token (4,994
chunks). The 225 → 110 ms drop is real, but not all of it is the engine.

**6. One request at a time still wastes the GPU.** Goodput at concurrency 4
and 16 was 1 request in 32 for every precision, same as the baseline. At
concurrency 16, p95 TTFT was still 60 s even for INT4. INT4 has room for 79k
tokens of KV cache but one request uses ~450 (295 in + 157 out on average). Phase 2 (batching) is what
uses that room.

**7. Cost per token fell 3.1x from the baseline, for 1.9 GSM8K points.**
$13.01 → $4.17 per 1M output tokens with INT4. FP8 is the safer middle:
$7.39 for 0.9 points (within noise). At INT4, the 16-bit `lm_head` is now
~26% of the bytes read per token, about 5 ms of the 20.8 ms step, so it's
the next target if quality allows.

---

## Piece 6a: FP8 KV cache

`--kv-cache-dtype fp8` stores each token's keys and values in 8 bits instead
of 16, halving the KV cache from 144 KB to 72 KB per token. The weights stay
the same (INT4, where the KV cache is the largest share of bytes read), so
the KV precision was meant to be the only variable. No calibrated KV scales
were used.

**It wasn't the only variable.** The server logs show vLLM picked the
FLASH_ATTN attention backend for every 16-bit KV run and FLASHINFER for every
FP8 KV run. With FP8 KV the candidate list drops to `['FLASHINFER',
'TRITON_ATTN']`, because FlashAttention can't read an FP8 KV cache on this
GPU. So the comparison below changes both the KV precision and the attention
kernel. To separate them I ran a control: 16-bit KV with FlashInfer forced,
on long_context. FlashAttention → FlashInfer (both 16-bit) isolates the
backend; FlashInfer 16-bit → FlashInfer FP8 isolates the KV precision.

```bash
lm_eval --model vllm \
  --model_args pretrained=$M,dtype=auto,max_model_len=4096,gpu_memory_utilization=0.85,kv_cache_dtype=fp8 \
  --tasks wikitext,gsm8k --batch_size auto --output_path optimize/evals/kv-fp8/
# chat: piece 5 settings. long_context: --max-model-len 16384, concurrency 1 only
# (at --max-num-seqs 1, higher levels only add queueing).
vllm serve $M --kv-cache-dtype fp8 ...   # vs. the default (16-bit KV)
```

The `long_context` prompts average 6,726 tokens (215,222 over 32 requests),
so the average context during decode is about 6,790 tokens.

### Results

| KV cache (INT4 weights) | Workload | TPOT p50 | Out tok/s | TTFT p50 | $ / 1M tokens |
| --- | --- | --- | --- | --- | --- |
| 16-bit | chat | 20.81 ms | 46.64 | 80 ms | $4.17 |
| FP8 | chat | 20.76 ms | 46.78 | 82 ms | $4.16 |
| 16-bit (FlashAttention) | long_context | 24.55 ms | 24.54 | 2,100 ms | $7.92 |
| 16-bit (FlashInfer, control) | long_context | 24.45 ms | 24.77 | 2,065 ms | $7.85 |
| FP8 (FlashInfer) | long_context | 22.78 ms | 25.77 | 2,076 ms | $7.54 |

| INT4 weights | WikiText word PPL | GSM8K strict |
| --- | --- | --- |
| 16-bit KV (piece 4) | 13.242 | 85.75 ± 0.96 |
| FP8 KV | 13.255 (+0.10%) | 86.73 ± 0.93 |

KV cache room from the server logs (INT4 weights, 85% memory):

| | 16-bit KV (FLASH_ATTN) | FP8 KV (FLASHINFER) |
| --- | --- | --- |
| Speed runs (`--max-num-seqs 1`) | 86,016 tokens | 165,888 tokens (1.93x) |
| Eval runs (default 256 seqs) | 79,360 tokens | 147,456 tokens (1.86x) |

### Findings

**1. The decode saving matched the bytes almost exactly.** At 6,790 tokens
of context the KV cache is 1.00 GB per step in 16-bit and 0.50 GB in FP8.
Model math predicted a 1.7 ms saving. The control splits the measured
1.77 ms into 0.10 ms from the backend switch (24.55 → 24.45 ms) and
**1.67 ms from the KV precision** (24.45 → 22.78 ms, −6.8%). The saved
0.50 GB in 1.67 ms is ~300 GB/s: every byte removed from the KV cache came
off the step at full memory speed. Decode attention is memory-bound too.

**2. On short chat prompts it made no difference, as predicted.** At ~450
tokens of context the KV cache is ~66 MB of the ~4.8 GB read per step (~1%),
so halving it should save ~0.1 ms. I measured 20.81 → 20.76 ms, inside the
noise.

**3. Long context costs what the bytes say.** Going from chat to
long_context with a 16-bit KV cache added 3.7 ms per step for ~0.9 GB more
KV read, about 250 GB/s. At 6.8k tokens the KV cache is already ~17% of
the bytes per step for INT4 weights.

**4. FP8 KV didn't change prefill.** With the backend held fixed, TTFT went
2,065 → 2,076 ms (+0.5%); the 35 ms drop in the uncontrolled comparison came
from FlashInfer, not from FP8. The cache is written once during prefill, and
the attention math still runs in 16-bit, so there's nothing to save. Prefill ran at ~3,200 tokens/s, about
51 TFLOPS counting attention, or ~43% of the L4's 16-bit peak.

**5. A 1-second TTFT is out of reach for 6.7k-token prompts on one L4.** The
prefill is ~107 TFLOPs of work. Even at the full 121 TFLOPS peak that takes
0.88 s, before any overhead, so goodput was 0 for both runs. Only fewer FLOPs
(FP8 tensor cores, which double the peak) or more GPUs can fix this, not
batching or scheduling.

**6. The quality cost is too small to measure, and GSM8K showed me its noise
floor.** Perplexity rose 0.10%. Perplexity is deterministic, so this is real
but tiny. GSM8K went *up* by 0.98 points. A change that can only add error
can't really improve accuracy, so ~1 point is the noise level of GSM8K at
1,319 problems. FP8 weights' −0.91 points in piece 4 is the same size, which
is why I called it indistinguishable from BF16.

**7. KV room nearly doubled, not exactly.** 86,016 → 165,888 tokens is
1.93x, not 2x. At 72 KB per token, the missing ~6k tokens are ~0.44 GB,
likely FlashInfer's workspace buffers (not verified). `--max-model-len` didn't
change KV room (86,016 at both 4,096 and 16,384); it only changes how many
full-length requests fit.

**8. Serving with `--max-num-seqs 1` leaves ~0.8–1.0 GB more for KV than the
evals did.** Same weights, same 85%: 15,872 / 62,464 / 86,016 tokens in the
piece 5 speed runs versus 10,240 / 56,320 / 79,360 in the piece 4 evals. vLLM
sizes the KV cache after profiling a dummy batch, and a 1-sequence limit
makes that profiled peak smaller. Phase 2 raises `--max-num-seqs` and will
give some of this back.

**9. FP8 KV is close to free, and its real payoff is capacity.** 6.8%
faster decode at long context from the KV precision alone, no measurable
quality cost, and about twice as many tokens in the same memory. That last
part only matters once many requests share the cache (Phase 2).

---

## Piece 6b: Speculative decoding

At one request at a time, a decode step reads all the weights to produce one
token. Checking several tokens in one forward pass reads the same weights
once, so it costs almost the same. Speculative decoding uses a cheap
**drafter** to guess the next *k* tokens. The target model then checks all
of them in one pass and keeps the longest correct prefix, plus one token of
its own. With greedy decoding the output is the same as normal decoding; only
the speed changes.

```
TPOT ≈ (T_target + T_verify_extra + k × T_draft) / L
L = mean acceptance length = tokens emitted per target pass (1 … k+1)
```

Drafters, both with k = 3 and INT4 target weights, each compared against the
piece 5 INT4 run (the only change is `--speculative-config`):

- **EAGLE-3** ([RedHatAI/Qwen3-8B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3)):
  one extra decoder layer that predicts from the target's own hidden states,
  with a 32k-token draft vocabulary, in BF16. It was trained on BF16 Qwen3-8B
  with thinking off. Predicted cost: ~0.8 GB read per draft token, ~2.7 ms.
- **n-gram** (prompt lookup): proposes the tokens that followed the last
  2–4 tokens wherever they appeared earlier in the prompt or output. It runs on
  the CPU, so drafting is close to free.

```bash
--speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-8B-speculator.eagle3","num_speculative_tokens":3}'
--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```

Acceptance comes from vLLM's Prometheus counters (`/metrics`) read after each
run, so it covers all three concurrency levels plus the 2 warmup requests.

**Caveat:** the chat workload is synthetic (sentences drawn from a small
pool), and `ignore_eos` makes the model keep generating past its natural
end. The acceptance rates here are comparable between drafters on this
workload, not representative of real chat traffic.

### Results (INT4 weights, concurrency 1)

| Drafter | L | Draft tokens accepted | TPOT p50 | TPOT mean | Out tok/s | TTFT p50 | $ / 1M tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| None (piece 5) | 1 | | 20.81 ms | 20.88 ms | 46.64 | 80 ms | $4.17 |
| n-gram, k=3 | 1.27 | 9.1% | 17.88 ms | 16.57 ms | 58.22 (1.25x) | 85 ms | $3.34 |
| EAGLE-3, k=3 | 2.35 | 45.1% | 13.15 ms | 12.69 ms | 74.80 (1.60x) | 91 ms | $2.60 |

TPOT mean = total decode time ÷ total tokens after the first, the average
that throughput depends on.

Acceptance by draft position (share of all drafts where that position was
accepted; a position only counts if every earlier one was also accepted):

| Drafter | 1st | 2nd | 3rd | 2nd given 1st | 3rd given 2nd |
| --- | --- | --- | --- | --- | --- |
| n-gram | 11.7% | 8.7% | 6.8% | 75% | 77% |
| EAGLE-3 | 65.7% | 43.1% | 26.5% | 66% | 61% |

KV cache room: 86,016 tokens without a drafter, 85,504 with n-gram,
68,608 with EAGLE-3.

### Findings

**1. EAGLE-3 made decode 1.6x faster, and the formula predicted it.** Mean
TPOT fell from 20.88 to 12.69 ms. Each speculative step took 29.9 ms
(12.69 × 2.35) and produced 2.35 tokens on average. Plugging the measured
L into my prediction: (20.9 + 3 × 2.7) ÷ 2.35 ≈ 12.3 ms, within 3% of the
measurement. Cost per 1M tokens fell from $4.17 to $2.60.

**2. Checking extra tokens is almost free; drafting them isn't.** The n-gram
drafter costs nothing on the GPU, so its step time shows the cost of checking
4 tokens instead of 1. That came to 21.1 ms vs 20.9 ms, only ~0.2 ms more.
Decode is memory-bound, so the extra math hides behind the weight reads. That
leaves 29.9 − 21.1 = 8.8 ms for EAGLE-3's three draft passes, 2.9 ms each,
against the 2.7 ms I predicted from bytes. So 29% of each speculative step
goes to drafting.

**3. k = 3 is about right for this target.** The 4th draft token would
cost ~2.9 ms. To pay for itself it would need to be accepted on more than
~23% of drafts. Position 3 was accepted 26.5% of the time, and each position
keeps ~61–66% of the previous one, so position 4 would land near 16%. k=4
would be slightly slower. Using the measured positions, k=2 would give
~12.9 ms, about the same as k=3. (k=4 is extrapolated, not measured.)

**4. EAGLE-3 reached its published acceptance even on synthetic text and an
INT4 target.** L = 2.35 at k = 3 sits inside the 2.13–2.48 Red Hat reports
on real tasks (with a BF16 target and sampling at temperature 0.6). Greedy
decoding helps acceptance, and the INT4 hidden states didn't visibly hurt
it. Without an FP16 run on the same workload, I can't measure any INT4
penalty directly.

**5. n-gram drafting is all-or-nothing.** Its first guess was right only
11.7% of the time, but once one guess was right the next ones were ~75%
right. It wins in bursts where the model copies text it has already seen, and
the synthetic prompts (repeated sentences from a small pool) give it a few
such bursts. TPOT per request ranged from 9.6 to 20.7 ms. It gave 1.25x
throughput at no memory cost, but this workload flatters it. Real
chat copies less, while summarization and code editing copy more.

**6. EAGLE-3 costs memory and a little TTFT.** KV room fell from 86,016 to
68,608 tokens. That's ~2.4 GB, consistent with the whole 2.04 GB draft
checkpoint (including its own copy of the 1.24 GB embedding table) plus a 37th
layer of KV cache. TTFT rose ~10 ms because the drafter also has to process
the prompt.

**7. The FP16 + EAGLE-3 run didn't start, and I didn't retry it.** At 85%
memory, FP16 had only 15,872 tokens (~2.2 GB) of KV room, less than the
~2.4 GB EAGLE-3 takes. A retry at 90% would test a prediction I'm leaving
open: the same ~8.8 ms of drafting is ~13% of a 60 ms FP16 step instead of
29% of an INT4 step, so FP16 should gain more (~2x if L stays ~2.35).


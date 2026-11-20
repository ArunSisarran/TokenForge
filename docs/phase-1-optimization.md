# Phase 1: Model-level optimization (in progress)

**Goal:** make each token cheaper at the model level (quantization, KV cache,
speculative decoding) and be able to explain *why* each change helps.

**Approach:** predict from first principles before measuring, then change
one variable at a time.

| Piece | Status |
| --- | --- |
| 1. Model math: memory, KV cache and decode ceilings per precision | ✅ done |
| 2. Profiling: where prefill and decode time goes | ✅ done |
| 3. FP8 and INT4 quantization | ✅ done |
| 4. Quality evals (perplexity, task accuracy) | ⬜ next |
| 5. Speed benchmarks: FP16 vs FP8 vs INT4 in vLLM | ⬜ |
| 6. FP8 KV cache and speculative decoding | ⬜ |

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

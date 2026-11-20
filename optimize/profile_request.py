"""
TokenForge — Phase 1: profile one request.

Runs prefill and decode manually (what generate() does internally), times
each phase, converts the times into hardware efficiency, and records a GPU
trace you can open at https://ui.perfetto.dev

Run from the repo root, on a GPU:
    python -m optimize.profile_request --model Qwen/Qwen3-8B
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimize.model_math import count_params, kv_bytes_per_token


def make_prompt_ids(tokenizer, n_tokens: int, device: str):
    # Content doesn't matter for speed, only length. Repeat a sentence and cut
    # it to exactly n_tokens.
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 5 + 10)
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to(device)


@torch.inference_mode()
def prefill(model, ids):
    # One forward pass over the WHOLE prompt. It fills the KV cache and gives
    # the logits for the next token. logits_to_keep=1: only compute the scores
    # for the last position, the only one we need.
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    next_tok = out.logits[:, -1:].argmax(dim=-1)
    return next_tok, out.past_key_values


@torch.inference_mode()
def decode_step(model, tok, past):
    # One forward pass over ONE new token, reusing the KV cache of everything
    # before it. This is repeated once per generated token.
    out = model(input_ids=tok, past_key_values=past, use_cache=True)
    return out.logits[:, -1:].argmax(dim=-1), out.past_key_values

def timed(fn):
    # GPU work is ASYNCHRONOUS: Python queues kernels and moves on immediately.
    # synchronize() waits until the GPU has really finished, so we time the
    # work and not just the queueing.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=64,
                    help="decode steps to time")
    ap.add_argument("--profile-steps", type=int, default=8,
                    help="decode steps to include in the trace (keeps the file small)")
    ap.add_argument("--bandwidth-gbs", type=float, default=300.0)
    ap.add_argument("--trace", default="optimize/traces/trace.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "profiling needs a GPU"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).to("cuda").eval()

    p = count_params(model.config)
    head = p["lm_head"] or p["embed"]
    kv_tok = kv_bytes_per_token(p)
    ids = make_prompt_ids(tokenizer, args.prompt_tokens, "cuda")
    T = ids.shape[1]

    # --- Warmup: the first passes pay one-time costs (kernel selection,
    # memory allocation). Run a few and throw the timings away.
    nxt, past = prefill(model, ids)
    for _ in range(5):
        nxt, past = decode_step(model, nxt, past)

    # --- Time prefill (fresh cache each time), take the median of 3.
    prefill_times = []
    for _ in range(3):
        _, t = timed(lambda: prefill(model, ids))
        prefill_times.append(t)
    t_prefill = statistics.median(prefill_times)

    # --- Time decode: one prefill, then many single-token steps.
    nxt, past = prefill(model, ids)
    step_times = []
    for _ in range(args.decode_steps):
        (nxt, past), t = timed(lambda: decode_step(model, nxt, past))
        step_times.append(t)
    t_step = statistics.median(step_times)

    # --- Turn times into hardware efficiency.
    # A matmul does 2 FLOPs (multiply + add) per weight per token.
    # Prefill runs every block weight over T tokens, and lm_head over 1 token.
    flops = 2 * p["blocks"] * T + 2 * head
    # A decode step reads every block weight, the lm_head, and the KV cache.
    ctx = T + args.decode_steps // 2   # average context length during timing
    bytes_step = p["blocks"] * 2 + head * 2 + ctx * kv_tok
    gbs = bytes_step / t_step / 1e9

    print(f"\nPrefill  {T} tokens in {t_prefill * 1000:.1f} ms"
          f"  -> {T / t_prefill:,.0f} tok/s, {flops / t_prefill / 1e12:.1f} TFLOPS achieved")
    print(f"Decode   median step {t_step * 1000:.1f} ms"
          f"  -> {1 / t_step:.1f} tok/s, {gbs:.0f} GB/s achieved"
          f" ({gbs / args.bandwidth_gbs:.0%} of {args.bandwidth_gbs:.0f} GB/s)")
    print(f"Ratio    one decode token costs {t_step / (t_prefill / T):,.0f}x one prefill token")
    print(f"Memory   peak allocated {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")

    # --- Record a trace of one prefill + a few decode steps.
    # record_function puts a named block on the timeline. synchronize() inside
    # each block makes the block's length match the GPU work it launched.
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("prefill"):
            nxt, past = prefill(model, ids)
            torch.cuda.synchronize()
        for _ in range(args.profile_steps):
            with record_function("decode_step"):
                nxt, past = decode_step(model, nxt, past)
                torch.cuda.synchronize()

    # Which GPU kernels took the most time?
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    prof.export_chrome_trace(args.trace)
    print(f"Trace saved to {args.trace}. Open it at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
"""
TokenForge — load generator / benchmark harness.

Sends a fixed workload to any OpenAI-compatible server at several concurrency
levels and records how fast every request was. The same script will measure
the Phase 0 baseline, vLLM in Phase 2, and the Kubernetes deployment later —
so it only relies on the OpenAI API, never on server internals.

Example:
    python bench/loadgen.py --url http://localhost:8000 \
        --workload bench/workloads/chat.jsonl \
        --concurrency 1 4 16 --num-requests 32 \
        --gpu-cost-per-hour 0.39 --label baseline-hf-fp16

THE METRICS (all per request, then summarized as percentiles):

  TTFT  time to first token: request sent -> first text arrives.
        Dominated by queueing + PREFILL. What a user feels as "is it alive?"
  TPOT  time per output token after the first:
        (end-to-end latency - TTFT) / (output tokens - 1).
        Dominated by DECODE. What a user feels as "reading speed".
  E2E   end-to-end latency: request sent -> last byte.

  and per concurrency level:

  output throughput  total output tokens / wall-clock time. Capacity.
  goodput            requests per second that met the SLO. The honest number:
                     throughput achieved by making users wait 30s is useless.
  $ / 1M tokens      GPU price per hour / tokens per hour * 1,000,000.
"""

import argparse
import asyncio
import json
import platform
import random
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------
async def send_one(client: httpx.AsyncClient, url: str, model: str | None,
                   item: dict, ignore_eos: bool) -> dict:
    """
    Send one STREAMING request and timestamp what happens.

    Why always stream? Without streaming we only learn when the whole answer
    is done, so TTFT is invisible. Streaming lets us see the moment the first
    token arrives, separating prefill time from decode time.
    """
    body = {
        "messages": item["messages"],
        "max_tokens": item["max_tokens"],
        # temperature 0 = greedy decoding: deterministic, so runs are repeatable.
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if model:
        body["model"] = model
    if ignore_eos:
        body["ignore_eos"] = True

    record = {"id": item["id"], "ok": False, "error": None,
              "ttft_s": None, "e2e_s": None, "tpot_s": None,
              "prompt_tokens": None, "completion_tokens": None,
              "num_chunks": 0}

    # perf_counter is a monotonic, high-resolution clock made for measuring
    # durations. time.time() can jump if the system clock is adjusted.
    t_start = time.perf_counter()
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")[:200]
                record["error"] = f"HTTP {resp.status_code}: {text}"
                return record
            # SSE: each event is a line "data: {...}"; blank lines separate them.
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    record["prompt_tokens"] = chunk["usage"]["prompt_tokens"]
                    record["completion_tokens"] = chunk["usage"]["completion_tokens"]
                choices = chunk.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    if record["ttft_s"] is None:
                        record["ttft_s"] = time.perf_counter() - t_start
                    record["num_chunks"] += 1
    except Exception as e:  # timeouts, refused connections, dropped streams
        record["error"] = repr(e)
        return record

    record["e2e_s"] = time.perf_counter() - t_start
    if record["ttft_s"] is None:
        record["error"] = "no content received"
        return record

    # If the server didn't report usage, fall back to counting chunks.
    # This is an approximation (a chunk may hold several tokens), so we flag it.
    if record["completion_tokens"] is None:
        record["completion_tokens"] = record["num_chunks"]
        record["tokens_estimated"] = True

    n = record["completion_tokens"]
    if n > 1:
        record["tpot_s"] = (record["e2e_s"] - record["ttft_s"]) / (n - 1)
    record["ok"] = True
    return record


# ---------------------------------------------------------------------------
# One concurrency level
# ---------------------------------------------------------------------------
async def run_level(url, model, items, concurrency, ignore_eos, timeout_s):
    """
    CLOSED-LOOP load: exactly `concurrency` virtual users. Each sends a
    request, waits for the full answer, then immediately sends the next one.
    So there are always `concurrency` requests in flight.

    (The alternative, OPEN-LOOP load, sends requests at a fixed rate whether or
    not earlier ones finished — closer to real internet traffic, and it can
    overwhelm a server. vLLM's `vllm bench serve --request-rate` does that.
    Closed-loop is easier to reason about, so we start with it.)
    """
    queue = iter(items)
    results = []

    # One shared HTTP client: reuses TCP connections instead of opening a new
    # one per request, so we measure the server and not connection setup.
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:

        async def virtual_user():
            for item in queue:  # every user pulls from the same iterator
                results.append(await send_one(client, url, model, item, ignore_eos))

        t0 = time.perf_counter()
        await asyncio.gather(*(virtual_user() for _ in range(concurrency)))
        wall_s = time.perf_counter() - t0

    return results, wall_s


async def warmup(url, model, items, ignore_eos, timeout_s, n):
    """
    The first requests to a fresh server are slow for reasons that have
    nothing to do with steady-state performance: CUDA kernels compile, memory
    pools allocate, caches are cold. Send a few and throw the timings away.
    """
    if n <= 0:
        return
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0)) as client:
        for item in items[:n]:
            await send_one(client, url, model, item, ignore_eos)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def percentile(values, p):
    """
    Why percentiles and not the average? Latency is skewed: a few slow
    requests barely move the mean but ruin some users' experience. p50 is the
    typical user; p95/p99 are the unlucky ones. SLOs are written on p95/p99.
    """
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(results, wall_s, concurrency, args):
    ok = [r for r in results if r["ok"]]
    ttft = [r["ttft_s"] for r in ok]
    tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    e2e = [r["e2e_s"] for r in ok]
    out_tokens = sum(r["completion_tokens"] for r in ok)
    in_tokens = sum(r["prompt_tokens"] or 0 for r in ok)

    def meets_slo(r):
        if args.slo_ttft_ms and r["ttft_s"] * 1000 > args.slo_ttft_ms:
            return False
        if args.slo_tpot_ms and r["tpot_s"] is not None and r["tpot_s"] * 1000 > args.slo_tpot_ms:
            return False
        return True

    good = [r for r in ok if meets_slo(r)]
    out_tps = out_tokens / wall_s if wall_s else 0

    summary = {
        "concurrency": concurrency,
        "requests": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": wall_s,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "output_tok_per_s": out_tps,
        "request_per_s": len(ok) / wall_s if wall_s else 0,
        "goodput_req_per_s": len(good) / wall_s if wall_s else 0,
        "slo_attainment": len(good) / len(results) if results else 0,
    }
    for name, vals in (("ttft_ms", ttft), ("tpot_ms", tpot), ("e2e_ms", e2e)):
        for p in (50, 95, 99):
            val = percentile(vals, p)
            summary[f"{name}_p{p}"] = val * 1000 if val is not None else None

    if args.gpu_cost_per_hour and out_tps > 0:
        tokens_per_hour = out_tps * 3600
        summary["usd_per_1m_output_tokens"] = args.gpu_cost_per_hour / tokens_per_hour * 1e6
    return summary


# ---------------------------------------------------------------------------
# Metadata: what exactly was measured?
# ---------------------------------------------------------------------------
def gather_metadata(args):
    """
    A number without context is useless six weeks later. Record everything
    that could explain a difference between two runs.
    """
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "client_host": socket.gethostname(),
        "python": platform.python_version(),
    }
    try:
        meta["server_health"] = httpx.get(args.url.rstrip("/") + "/health", timeout=5).json()
    except Exception as e:
        meta["server_health"] = f"unavailable: {e!r}"
    for key, cmd in {
        "git_commit": ["git", "rev-parse", "--short", "HEAD"],
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"],
    }.items():
        try:
            meta[key] = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=5).stdout.strip() or None
        except Exception:
            meta[key] = None
    return meta


def fmt(v, spec=".0f"):
    return "-" if v is None else format(v, spec)


def print_table(rows):
    header = (f"{'conc':>5} {'ok/total':>9} {'out tok/s':>10} {'TTFT p50':>9} "
              f"{'TTFT p95':>9} {'TPOT p50':>9} {'TPOT p95':>9} {'goodput':>8} {'$/1M tok':>9}")
    print(header)
    print("-" * len(header))
    for s in rows:
        print(f"{s['concurrency']:>5} {s['succeeded']:>4}/{s['requests']:<4} "
              f"{fmt(s['output_tok_per_s'], '.1f'):>10} "
              f"{fmt(s['ttft_ms_p50']):>8}ms {fmt(s['ttft_ms_p95']):>7}ms "
              f"{fmt(s['tpot_ms_p50'], '.1f'):>7}ms {fmt(s['tpot_ms_p95'], '.1f'):>7}ms "
              f"{fmt(s['goodput_req_per_s'], '.2f'):>8} "
              f"{fmt(s.get('usd_per_1m_output_tokens'), '.2f'):>9}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    p = argparse.ArgumentParser(description="TokenForge benchmark harness")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--workload", required=True, help="JSONL file from make_workloads.py")
    p.add_argument("--model", default=None, help="model name to send (vLLM requires it)")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--num-requests", type=int, default=32, help="requests per level")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ignore-eos", action="store_true",
                   help="force every request to generate exactly max_tokens")
    p.add_argument("--max-tokens-cap", type=int, default=None,
                   help="cap max_tokens per request (keeps slow baselines short)")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds")
    p.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=100.0)
    p.add_argument("--gpu-cost-per-hour", type=float, default=None,
                   help="what the GPU costs you per hour, for $/1M tokens")
    p.add_argument("--label", default="run", help="name for this run, e.g. baseline-hf-fp16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="bench/results")
    args = p.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    items = [json.loads(line) for line in Path(args.workload).read_text().splitlines() if line.strip()]
    if args.max_tokens_cap:
        for it in items:
            it["max_tokens"] = min(it["max_tokens"], args.max_tokens_cap)

    # Shuffle with a fixed seed: a fair mix of short and long requests in each
    # level, but the same order every time you rerun.
    random.Random(args.seed).shuffle(items)
    if len(items) < args.num_requests:
        print(f"note: workload has only {len(items)} requests")

    print(f"Benchmarking {endpoint} with {Path(args.workload).name}, label={args.label}")
    meta = gather_metadata(args)
    await warmup(endpoint, args.model, items, args.ignore_eos, args.timeout, args.warmup)

    summaries, raw = [], {}
    for c in args.concurrency:
        level_items = items[: args.num_requests]
        print(f"  concurrency={c}: {len(level_items)} requests ...", flush=True)
        results, wall_s = await run_level(endpoint, args.model, level_items, c,
                                          args.ignore_eos, args.timeout)
        s = summarize(results, wall_s, c, args)
        summaries.append(s)
        raw[str(c)] = results  # keep every request, not just the summary
        if s["failed"]:
            first_err = next(r["error"] for r in results if not r["ok"])
            print(f"    {s['failed']} failed, e.g. {first_err}")

    print()
    print_table(summaries)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{args.label}-{Path(args.workload).stem}-{stamp}.json"
    out_path.write_text(json.dumps({"metadata": meta, "summary": summaries, "requests": raw}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
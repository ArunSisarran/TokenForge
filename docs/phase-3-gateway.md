# Phase 3: Gateway

**Goal:** put a gateway in front of the engine that handles what vLLM
doesn't: API keys, per-key rate limits and safe input. It must not slow
streaming down.

**Approach:** build the smallest gateway that does the job, test it without a
GPU, then measure what it costs by running the same load directly and through
the gateway.

**Summary:**

| | Result |
| --- | --- |
| TTFT added | +2.4 ms at c=1, +3.1 ms at c=64 (steady-state p50) |
| TPOT added | none measurable (within 0.05 ms at p50, c=1 to 256) |
| Throughput | unchanged up to c=64; −7.8% at c=256 with one process |
| Where it strains | c=256, one process: steady-state TTFT p95 522 ms vs 102 ms direct |
| Headroom | one process forwarded 7,385 tokens/s, ~8x one fully loaded L4 (954 tok/s) |

| Piece | Status |
| --- | --- |
| 1. Gateway: keys, input checks, streaming passthrough, timeouts | ✅ done |
| 2. Rate limits: Redis token buckets (requests and tokens per minute) | ✅ done |
| 3. Overhead benchmark: direct vs through the gateway | ✅ done |

---

## Design (`gateway/app.py`)

```bash
UPSTREAM_URL=http://localhost:8000 REDIS_URL=redis://localhost:6379 \
    uvicorn gateway.app:app --port 8080
python -m gateway.keys create alice --rpm 60 --tpm 20000   # prints the key once
```

I wrote it in Python with FastAPI, the stack the rest of the project already
uses. Piece 3 measures whether Python is fast enough, rather than assuming Go
is needed.

What happens to one `POST /v1/chat/completions`:

1. **Authentication.** The bearer key is hashed with SHA-256 and looked up as
   `key:<hash>` in Redis, which holds the key's name and limits. Redis never
   stores the key itself, so a dump of Redis leaks nothing usable. A slow
   password hash isn't needed: the keys are 256 random bits, not
   human-chosen passwords.
2. **Input checks.** The body must be under 1 MB and a JSON object.
   `max_tokens` (or `max_completion_tokens`) must be an integer from 1 to 4096
   and is set to 512 when missing. Without a limit, vLLM would generate up to
   the full context length, and the rate limiter needs a number to charge. `n`
   must be 1–8.
3. **Rate limits.** Each key has two token buckets: requests per minute and
   output tokens per minute. Each refills at limit ÷ 60 per second and holds
   at most one minute's worth. A request costs 1 request and
   `max_tokens × n` tokens. One Lua script runs inside Redis to check and
   charge both buckets:
   - **Atomic:** Redis runs a script without interruption. Several gateway
     processes can't both read "1 left" and both let a request through.
   - **All or nothing:** a request refused on tokens doesn't use up a request
     slot.
   - **Redis's clock** (`TIME`), so gateway machines with slightly different
     clocks still agree.
   - **Retry-After:** when a request doesn't fit, the client gets 429 with
     `Retry-After` set to the seconds until it would.
   - **Self-cleaning:** idle buckets expire after 120 s. A missing bucket
     counts as full, which is what it would be by then anyway.
4. **Streaming passthrough.** The gateway opens the engine response with
   `send(stream=True)` and forwards each chunk as it arrives, without
   buffering. It uses `aiter_bytes`, which decompresses, so a compressed
   engine response could never reach the client garbled.
5. **Disconnects.** When the client drops mid-stream, the gateway closes the
   engine connection, so vLLM aborts the request and frees its batch slot and
   KV cache. I checked this with real processes: the engine connection was
   open during the stream and gone within a second after the client
   disconnected.

Other choices:

- **Timeouts:** connect 5 s (fail fast with 502), read 120 s *between
  chunks*, no total. Long streams are normal, and Phase 2's overloaded runs
  had TTFTs around 30 s.
- **Connection pool of 1,000.** httpx's default of 100 would make user 101
  silently wait for a connection, which would only show up as TTFT.
- **Fail closed.** If Redis is down, requests get 503 rather than unlimited
  access to the GPU.
- **OpenAI error format** (`{"error": {...}}`) for every status (401, 400,
  413, 429, 502, 503, 504), so OpenAI clients show a proper message.
- **`/health` checks only that the gateway process is up.** If it also
  checked the engine, Kubernetes would restart healthy gateways every time
  vLLM restarted (Phase 5).

**Known shortcut:** tokens are charged at `max_tokens` up front and never
refunded, and prompt tokens aren't counted. Short replies use up a key's
token limit faster than they should. The fix is to refund the difference
using the usage numbers in the final chunk.

**Not built:** `/v1/completions`, usage accounting per key, a key from the
gateway to vLLM (that link stays on a private network), and a readiness
check (Phase 5).

## Testing without a GPU

- `gateway/fake_upstream.py` is a fake engine with a fixed TTFT and TPOT
  (`TTFT_MS`, `TPOT_MS`) that streams in vLLM's format. The tests use it, and
  so does the overhead benchmark.
- `gateway/test_gateway.py` has 5 tests that run in-process against
  fakeredis, with no server needed (`python -m pytest gateway -q`). They
  cover authentication, input checks, passthrough including the usage chunk,
  429 with `Retry-After`, a refused request not charging either bucket, and
  502 when the engine is down. To check that the atomicity test can fail, I
  broke the Lua so a refused request still charged its request slot, and the
  test failed.

---

## Piece 3: Overhead

The fake engine was set to INT4's batch-1 timings (TTFT 80 ms, TPOT 21 ms).
I ran loadgen with the standard chat settings and `--min-rounds 4`, first
directly against the engine and then through one gateway process (one
uvicorn worker) with real Redis 8.12. Everything ran on one laptop (15 CPU
cores), so both runs shared the same machine and the comparison is fair. The
c=256 level sent 500 requests, the size of the chat workload, so it is
weighted toward the start burst.

| c | tok/s direct → gateway | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 44.7 → 44.7 | 85.4 → 87.7 ms | 85.8 → 88.8 ms | 21.95 → 21.96 ms | 22.00 → 21.99 ms |
| 16 | 574 → 573 | 86.1 → 88.7 ms | 94.4 → 106.2 ms | 21.92 → 21.96 ms | 21.97 → 22.04 ms |
| 64 | 2,386 → 2,385 | 89.6 → 92.8 ms | 113.7 → 130.7 ms | 21.95 → 21.91 ms | 22.02 → 21.96 ms |
| 256 | 8,007 → 7,385 | 124 → 289 ms | 164 → 494 ms | 22.00 → 22.05 ms | 22.06 → 22.75 ms |

As in Phase 2, I split TTFT into the first wave (all c users at once) and
steady state:

| c | First-wave TTFT p50 | Steady TTFT p50 | Steady TTFT p95 |
| --- | --- | --- | --- |
| 1 | 85 → 87 ms | 85.4 → 87.8 ms | 85.9 → 88.8 ms |
| 16 | 94 → 106 ms | 85.0 → 88.1 ms | 90.1 → 94.6 ms |
| 64 | 111 → 120 ms | 88.3 → 91.4 ms | 93.1 → 107.8 ms |
| 256 | 161 → 394 ms | 88.5 → 91.9 ms | 102 → 522 ms |

### Findings

**1. The gateway adds about 2.5 ms of TTFT per request** (+2.4 ms at c=1,
+3.1 ms at c=64, steady-state p50). That's two Redis round trips, parsing the
JSON and one extra local HTTP hop. Against vLLM's real TTFT on the L4 (77 ms
at c=1, 372 ms at c=64), it's under 3%.

**2. TPOT doesn't change** (within 0.05 ms at p50 at every level). Once a
request is admitted, the gateway only forwards bytes: microseconds per chunk
against a 21 ms gap between chunks.

**3. At c=256 one process strains, and it shows in TTFT, not TPOT.** Steady
TTFT p50 was still fine (91.9 vs 88.5 ms), but p95 rose to 522 ms vs 102 ms,
first-wave p50 to 394 vs 161 ms, and throughput fell 7.8%. One process runs a
single event loop. At c=256 that loop forwards ~7,400 chunks/s, and each new
request's work (auth, two Redis calls, JSON parsing, maybe a new engine
connection) waits its turn behind the forwarding. Requests already streaming
barely notice (TPOT p95 +0.7 ms). I didn't profile CPU use, so this
explanation comes from the latency pattern.

**4. One gateway process has ~8x the headroom of one L4.** The fastest Phase 2
config produced 954 tok/s at c=64, so that's ~954 chunks/s to forward. One
gateway process forwarded 7,385 chunks/s with steady TTFT p50 under 100 ms.
At this scale, Python vs Go doesn't matter. More capacity comes from running
more processes (`--workers`), and the Redis buckets keep the limits correct
across them. I didn't run the multi-worker case.

## What this sets up

- **Phase 4 (Docker):** the gateway, Redis and vLLM become one compose stack,
  and the gateway runs in front of real vLLM for the first time. So far it
  has only run in front of the fake engine.
- **Phase 5 (Kubernetes):** gateway replicas share limits through Redis,
  `/health` is the liveness probe, and a readiness check still needs adding.

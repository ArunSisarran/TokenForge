"""
TokenForge gateway: API keys, per-key rate limits and streaming passthrough
in front of an OpenAI-compatible engine (vLLM or SGLang).

Run (Redis and the engine must already be up):
    UPSTREAM_URL=http://localhost:8000 REDIS_URL=redis://localhost:6379 \
        uvicorn gateway.app:app --port 8080
"""

import hashlib
import json
import math
import os
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://localhost:8000").rstrip("/")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
MAX_BODY_BYTES = 1_000_000
DEFAULT_MAX_TOKENS = 512
MAX_TOKENS_CAP = 4096

# Token buckets, checked and charged atomically. KEYS: one bucket per limit.
# ARGV: (per-minute limit, cost) for each bucket. Charges every bucket or none.
# Returns "0" if allowed, else seconds until the request would fit.
RATE_LIMIT_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local level, wait = {}, 0
for i, key in ipairs(KEYS) do
  local cap, cost = tonumber(ARGV[2*i-1]), tonumber(ARGV[2*i])
  local b = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(b[1]) or cap
  local ts = tonumber(b[2]) or now
  tokens = math.min(cap, tokens + (now - ts) * cap / 60)
  if tokens < cost then wait = math.max(wait, (cost - tokens) * 60 / cap) end
  level[i] = tokens
end
if wait > 0 then return tostring(wait) end
for i, key in ipairs(KEYS) do
  redis.call('HSET', key, 'tokens', level[i] - tonumber(ARGV[2*i]), 'ts', now)
  redis.call('EXPIRE', key, 120)
end
return '0'
"""


class APIError(Exception):
    """An error returned to the client in OpenAI's error format."""

    def __init__(self, status: int, message: str, headers: dict | None = None):
        self.status, self.message, self.headers = status, message, headers


def key_hash(key: str) -> str:
    """Hash an API key; Redis stores only hashes, never the keys themselves."""
    return hashlib.sha256(key.encode()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open one Redis client and one pooled upstream HTTP client for the process."""
    app.state.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    app.state.http = httpx.AsyncClient(
        base_url=UPSTREAM_URL,
        # No total timeout: a stream can legitimately run for minutes.
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0),
        # httpx's default cap of 100 connections would silently queue user 101.
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000),
    )
    yield
    await app.state.http.aclose()
    await app.state.redis.aclose()


app = FastAPI(title="TokenForge gateway", lifespan=lifespan)


@app.exception_handler(APIError)
async def api_error(request: Request, e: APIError):
    """Render APIError as {"error": {...}} so OpenAI clients can parse it."""
    return JSONResponse({"error": {"message": e.message, "code": e.status}},
                        status_code=e.status, headers=e.headers)


async def authenticate(request: Request) -> tuple[str, dict]:
    """Look up the bearer key; return its hash and its limits."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise APIError(401, "missing API key (Authorization: Bearer <key>)")
    h = key_hash(auth.removeprefix("Bearer ").strip())
    try:
        info = await request.app.state.redis.hgetall(f"key:{h}")
    except redis.RedisError:
        raise APIError(503, "rate limiter unavailable")
    if not info:
        raise APIError(401, "invalid API key")
    return h, info


async def read_json(request: Request) -> dict:
    """Read the request body with a size cap and parse it as a JSON object."""
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise APIError(413, f"request body over {MAX_BODY_BYTES} bytes")
    try:
        data = json.loads(body)
    except ValueError:
        raise APIError(400, "body is not valid JSON")
    if not isinstance(data, dict):
        raise APIError(400, "body must be a JSON object")
    return data


def output_budget(body: dict) -> int:
    """Validate max_tokens and n, fill in a default max_tokens, return the most tokens this request can generate."""
    key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
    body.setdefault(key, DEFAULT_MAX_TOKENS)
    max_tokens, n = body[key], body.get("n", 1)
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS_CAP:
        raise APIError(400, f"{key} must be an integer from 1 to {MAX_TOKENS_CAP}")
    if type(n) is not int or not 1 <= n <= 8:
        raise APIError(400, "n must be an integer from 1 to 8")
    return max_tokens * n


async def charge(request: Request, h: str, info: dict, tokens: int):
    """Take 1 request and `tokens` output tokens from this key's buckets, or raise 429."""
    rpm, tpm = int(info["rpm"]), int(info["tpm"])
    if tokens > tpm:
        raise APIError(400, f"this request can generate {tokens} tokens, over your limit of {tpm} per minute")
    # shortcut: charges max_tokens up front and never refunds unused tokens or counts
    # prompt tokens; refund from the final usage chunk if TPM limits feel too strict.
    r = request.app.state.redis
    try:
        wait = float(await r.register_script(RATE_LIMIT_LUA)(
            keys=[f"rl:{h}:rpm", f"rl:{h}:tpm"], args=[rpm, 1, tpm, tokens]))
    except redis.RedisError:
        raise APIError(503, "rate limiter unavailable")
    if wait:
        raise APIError(429, f"rate limit exceeded, retry in {wait:.1f} s",
                       {"Retry-After": str(math.ceil(wait))})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Authenticate, rate-limit, then stream the engine's response through unchanged."""
    h, info = await authenticate(request)
    body = await read_json(request)
    await charge(request, h, info, output_budget(body))

    http = request.app.state.http
    try:
        resp = await http.send(http.build_request("POST", "/v1/chat/completions", json=body),
                               stream=True)
    except httpx.TimeoutException:
        raise APIError(504, "engine timed out")
    except httpx.HTTPError:
        raise APIError(502, "engine unavailable")
    # aclose runs when the stream ends or the client disconnects; closing the
    # upstream connection makes vLLM abort the request and free its KV cache.
    return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code,
                             media_type=resp.headers.get("content-type"),
                             background=BackgroundTask(resp.aclose))


@app.get("/v1/models")
async def models(request: Request):
    """Pass the engine's model list through to authenticated clients."""
    await authenticate(request)
    try:
        resp = await request.app.state.http.get("/v1/models")
    except httpx.HTTPError:
        raise APIError(502, "engine unavailable")
    return JSONResponse(resp.json(), status_code=resp.status_code)


@app.get("/health")
async def health():
    """Liveness only: the gateway process is up. It deliberately doesn't check the engine."""
    return {"status": "ok"}

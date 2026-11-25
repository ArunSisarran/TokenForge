"""
Fake OpenAI-compatible engine with fixed timings, for testing and benchmarking
the gateway without a GPU. Always streams, in vLLM's SSE format.

Run:
    TTFT_MS=50 TPOT_MS=20 uvicorn gateway.fake_upstream:app --port 8000
"""

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

TTFT_S = float(os.environ.get("TTFT_MS", "50")) / 1000
TPOT_S = float(os.environ.get("TPOT_MS", "20")) / 1000

app = FastAPI(title="TokenForge fake engine")


def sse(obj) -> str:
    """Format one server-sent event."""
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Stream max_tokens one-word chunks: TTFT_MS before the first, TPOT_MS between the rest."""
    body = await request.json()
    n = body.get("max_tokens") or body.get("max_completion_tokens") or 16
    prompt = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))

    async def stream():
        await asyncio.sleep(TTFT_S)
        for i in range(n):
            if i:
                await asyncio.sleep(TPOT_S)
            yield sse({"object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": {"content": "tok "}}]})
        if body.get("stream_options", {}).get("include_usage"):
            yield sse({"object": "chat.completion.chunk", "choices": [],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": n,
                                 "total_tokens": prompt + n}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/v1/models")
async def models():
    """List the one fake model."""
    return {"object": "list", "data": [{"id": "fake", "object": "model"}]}


@app.get("/health")
async def health():
    """Liveness check."""
    return {}

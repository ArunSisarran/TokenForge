"""
TokenForge Phase 0: naive Hugging Face server, one request at a time.
Frozen on purpose: it is the "before" measurement. Do not optimize it.

Run:
    MODEL_ID=Qwen/Qwen3-0.6B uvicorn baseline.server:app --port 8000
"""

import json
import os
import threading
import time
import uuid
from typing import Any
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CPUs are slow at float16 math, so fall back to float32 there.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
_t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
model.eval()
print(f"Loaded in {time.time() - _t0:.1f}s")

PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# One generate() at a time: the bottleneck this baseline exists to measure.
GPU_LOCK = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False
    ignore_eos: bool = False
    stream_options: dict | None = None


app = FastAPI(title="TokenForge baseline")


@app.get("/health")
def health():
    """Liveness check. Never touches the model."""
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


def build_inputs(messages: list[Message]):
    """Apply the chat template and tokenize."""
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # keeps output lengths comparable between runs
    )
    return tokenizer(prompt, return_tensors="pt").to(DEVICE)


def generation_kwargs(req: ChatRequest, inputs) -> dict:
    """Build generate() arguments from the request."""
    kwargs: dict[str, Any] = dict(**inputs, max_new_tokens=req.max_tokens, pad_token_id=PAD_ID)
    if req.ignore_eos:
        kwargs["min_new_tokens"] = req.max_tokens
    if req.temperature > 0:
        kwargs.update(do_sample=True, temperature=req.temperature)
    else:
        kwargs.update(do_sample=False)
    return kwargs


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat completion, streaming or not."""
    inputs = build_inputs(req.messages)
    prompt_tokens = inputs["input_ids"].shape[1]

    if req.stream:
        return StreamingResponse(
            stream_tokens(req, inputs, prompt_tokens), media_type="text/event-stream"
        )

    with GPU_LOCK:
        with torch.inference_mode():
            output = model.generate(**generation_kwargs(req, inputs))

    new_tokens = output[0, prompt_tokens:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if completion_tokens >= req.max_tokens else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_tokens(req: ChatRequest, inputs, prompt_tokens: int):
    """Stream the reply as SSE chunks; generate() runs in a thread feeding the streamer."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    result = {"completion_tokens": 0}  # filled in by the generate thread

    def chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    with GPU_LOCK:
        def run_generate():
            # inference_mode is per-thread, so it must be set inside the thread.
            with torch.inference_mode():
                output = model.generate(**generation_kwargs(req, inputs), streamer=streamer)
            result["completion_tokens"] = output.shape[1] - prompt_tokens

        worker = threading.Thread(target=run_generate)
        worker.start()
        try:
            yield chunk({"role": "assistant"})
            for text_piece in streamer:
                if text_piece:
                    yield chunk({"content": text_piece})
        finally:
            # Hold the lock until generate() ends, even if the client disconnected.
            worker.join()

    n = result["completion_tokens"]
    yield chunk({}, finish_reason="length" if n >= req.max_tokens else "stop")

    if req.stream_options and req.stream_options.get("include_usage"):
        usage_payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n,
                "total_tokens": prompt_tokens + n,
            },
        }
        yield f"data: {json.dumps(usage_payload)}\n\n"
    yield "data: [DONE]\n\n"

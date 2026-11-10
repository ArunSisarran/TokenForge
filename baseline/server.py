"""
TokenForge — Phase 0 baseline server.

PURPOSE: This server is intentionally naive. It serves an LLM the "obvious" way
(Hugging Face transformers + one request at a time) so that we have a baseline
to measure every later optimization against. Do not optimize this file — its
slowness is the point.

Run:
    MODEL_ID=Qwen/Qwen3-0.6B uvicorn baseline.server:app --port 8000
"""

import json
import os
import threading
import time
import uuid
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Model environment var
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load each param in float16 to take up less space on the GPU
# CPU are slow at float16 math so if its not run on the GPU use float32
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Load the model once at startup
print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
_t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()  # switches off training-only behavior such as dropout
print(f"Loaded in {time.time() - _t0:.1f}s")

PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# Lock allows model.generate() to handle one request at a time, each requests waits for the other to finish
GPU_LOCK = threading.Lock()

# Make the request schema OpenAI compatible
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


app = FastAPI(title="TokenForge baseline")


@app.get("/health")
def health():
    # Kubernetes will call an endpoint like this to decide whether the pod is
    # alive. It is cheap on purpose: it must not touch the model.
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


def build_inputs(messages: list[Message]):
    # Chat models were trained on a specific text layout with special tokens
    # marking who is speaking. apply_chat_template turns our list of messages
    # into exactly that layout. add_generation_prompt=True appends the
    # "assistant's turn starts here" marker so the model answers instead of
    # continuing the user's message. enable_thinking=False turns off Qwen3's
    # reasoning mode so output lengths stay comparable between runs.
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Tokenization: text -> integer IDs. The model only ever sees these IDs.
    return tokenizer(prompt, return_tensors="pt").to(DEVICE)


def generation_kwargs(req: ChatRequest, inputs) -> dict:
    kwargs = dict(**inputs, max_new_tokens=req.max_tokens, pad_token_id=PAD_ID)
    if req.temperature > 0:
        kwargs.update(do_sample=True, temperature=req.temperature)
    else:
        kwargs.update(do_sample=False)  # greedy: always pick the top token
    return kwargs


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    inputs = build_inputs(req.messages)
    prompt_tokens = inputs["input_ids"].shape[1]

    if req.stream:
        return StreamingResponse(
            stream_tokens(req, inputs), media_type="text/event-stream"
        )

    with GPU_LOCK:
        # inference_mode tells PyTorch not to record operations for
        # backpropagation. We never train here, so this saves memory and time.
        with torch.inference_mode():
            # What generate() does internally:
            #   PREFILL: run the whole prompt through the model in one pass.
            #            Compute-heavy; it decides time-to-first-token (TTFT).
            #   DECODE:  loop: predict ONE token, append it, repeat. Each step
            #            re-reads all the weights from GPU memory to produce a
            #            single token — memory-bandwidth-bound, and it decides
            #            inter-token latency. The KV cache stores attention
            #            keys/values from earlier tokens so they aren't
            #            recomputed every step.
            output = model.generate(**generation_kwargs(req, inputs))

    # generate() returns prompt + new tokens; slice off the prompt.
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
        # Token counts are how LLM work is measured and billed. The benchmark
        # will use them to compute tokens/sec and cost per 1M tokens.
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

# Streaming with Server-Sent Events (SSE)
# sends each piece of text as soon as it exists.

# generate() blocks until it finishes, so we run it in a background thread.
# TextIteratorStreamer is a queue: generate() pushes decoded text into it,
# and this function pulls text out and sends it to the client.
def stream_tokens(req: ChatRequest, inputs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

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
                model.generate(**generation_kwargs(req, inputs), streamer=streamer)

        worker = threading.Thread(target=run_generate)
        worker.start()
        try:
            yield chunk({"role": "assistant"})
            for text_piece in streamer:
                if text_piece:
                    yield chunk({"content": text_piece})
        finally:
            # Even if the client disconnects, wait for generate() to finish
            # before releasing the lock. (A real engine would cancel the
            # request and free its memory — another thing vLLM does for us.)
            worker.join()

    yield chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"

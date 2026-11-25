"""
Gateway tests: auth, validation, rate limits, passthrough and engine errors.
Runs in-process with fakeredis and the fake engine (no Redis server, no GPU).

Run:
    python -m pytest gateway -q
"""

import asyncio

import fakeredis
import httpx

from gateway import app as gw
from gateway import fake_upstream

fake_upstream.TTFT_S = fake_upstream.TPOT_S = 0
KEY = "tf-test-key"
CHAT = {"model": "fake", "messages": [{"role": "user", "content": "hi there"}],
        "max_tokens": 5, "stream": True, "stream_options": {"include_usage": True}}
URL = "/v1/chat/completions"


def down(request):
    """A transport for an engine that refuses connections."""
    raise httpx.ConnectError("connection refused", request=request)


async def gateway(rpm=60, tpm=10_000, engine=httpx.ASGITransport(app=fake_upstream.app)):
    """Fresh gateway state with one key; returns a client that sends that key."""
    gw.app.state.redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    await gw.app.state.redis.hset(f"key:{gw.key_hash(KEY)}",
                                  mapping={"name": "test", "rpm": rpm, "tpm": tpm})
    gw.app.state.http = httpx.AsyncClient(transport=engine, base_url="http://engine")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gateway",
                             headers={"Authorization": f"Bearer {KEY}"})


def test_auth_and_passthrough():
    async def t():
        c = await gateway()
        assert (await c.post(URL, json=CHAT, headers={"Authorization": ""})).status_code == 401
        assert (await c.post(URL, json=CHAT, headers={"Authorization": "Bearer nope"})).status_code == 401
        r = await c.post(URL, json=CHAT)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.text.count("tok ") == 5 and '"completion_tokens": 5' in r.text
        assert r.text.endswith("data: [DONE]\n\n")
        assert (await c.get("/v1/models")).json()["data"][0]["id"] == "fake"
    asyncio.run(t())


def test_validation():
    async def t():
        c = await gateway(tpm=1000)
        for bad in (0, -1, "5", 5.0, True, 5000):
            assert (await c.post(URL, json={**CHAT, "max_tokens": bad})).status_code == 400, bad
        assert (await c.post(URL, json={**CHAT, "n": 0})).status_code == 400
        assert (await c.post(URL, content=b"not json")).status_code == 400
        assert (await c.post(URL, json=[CHAT])).status_code == 400
        assert (await c.post(URL, content=b"x" * 1_000_001)).status_code == 413
        assert (await c.post(URL, json={**CHAT, "max_tokens": 600, "n": 2})).status_code == 400
        no_max = {k: v for k, v in CHAT.items() if k != "max_tokens"}
        r = await c.post(URL, json=no_max)
        assert f'"completion_tokens": {gw.DEFAULT_MAX_TOKENS}' in r.text
    asyncio.run(t())


def test_request_limit():
    async def t():
        c = await gateway(rpm=2)
        assert [(await c.post(URL, json=CHAT)).status_code for _ in range(2)] == [200, 200]
        r = await c.post(URL, json=CHAT)
        assert r.status_code == 429 and r.headers["Retry-After"] == "30"
        assert "rate limit" in r.json()["error"]["message"]
    asyncio.run(t())


def test_denied_request_charges_no_bucket():
    async def t():
        c = await gateway(rpm=3, tpm=100)
        codes = [(await c.post(URL, json={**CHAT, "max_tokens": m})).status_code
                 for m in (80, 80, 10, 10, 1)]
        # The 2nd request fails on tokens and must not use up a request slot,
        # so the 3rd and 4th still fit; the 5th is the 4th request to pass.
        assert codes == [200, 429, 200, 200, 429]
    asyncio.run(t())


def test_engine_down():
    async def t():
        c = await gateway(engine=httpx.MockTransport(down))
        r = await c.post(URL, json=CHAT)
        assert r.status_code == 502 and r.json()["error"]["message"] == "engine unavailable"
    asyncio.run(t())

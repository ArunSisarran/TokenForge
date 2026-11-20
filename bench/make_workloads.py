"""
TokenForge: generate the fixed, seeded benchmark workloads.
Never edit a committed workload; write a new file (e.g. chat_v2.jsonl) instead.

  chat.jsonl           mixed lengths, general serving
  shared_prefix.jsonl  same ~2k-token system prompt, for prefix caching
  long_context.jsonl   ~8k-token inputs, for prefill and KV cache memory

Run:
    python bench/make_workloads.py --out-dir bench/workloads
"""

import argparse
import json
import math
import random
from pathlib import Path

SENTENCES = [
    "The committee reviewed the quarterly figures before the meeting ended.",
    "Rainfall in the northern valley was higher than the ten year average.",
    "Engineers replaced the bearings after noticing unusual vibration.",
    "The library extended its opening hours during the exam period.",
    "Shipping delays affected several suppliers in the second half of the year.",
    "A new bus route now connects the hospital with the train station.",
    "The recipe calls for slow roasting at a low temperature for three hours.",
    "Volunteers planted four hundred trees along the river bank.",
    "The software update fixed a memory leak in the reporting module.",
    "Attendance at the museum rose after the new exhibit opened.",
    "The contract requires thirty days notice before termination.",
    "Researchers measured soil moisture at twelve sites every morning.",
    "The bridge will be closed to traffic for inspection next weekend.",
    "Customer complaints fell after the support team added evening shifts.",
    "The orchestra rehearsed the second movement twice before lunch.",
    "Battery capacity declined slowly over five hundred charge cycles.",
    "The city council approved a budget for repairing old sidewalks.",
    "Warehouse staff tested a new system for tracking returned items.",
    "The hiking trail climbs steadily for two miles before the summit.",
    "Analysts expect demand for the product to stay flat next quarter.",
]

QUESTIONS = [
    "Summarize the main points in three sentences.",
    "What are the risks described here, and how could they be reduced?",
    "Write a short email to a colleague explaining this.",
    "List the key facts as bullet points.",
    "Explain this to a ten year old.",
    "What questions would you ask to learn more?",
    "Rewrite this in a more formal tone.",
    "What is the most important detail, and why?",
    "Turn this into a short news headline and a one paragraph story.",
    "Give three possible next steps based on this information.",
]


def make_text(rng: random.Random, n_words: int) -> str:
    """Build roughly n_words of text by sampling sentences."""
    out, count = [], 0
    while count < n_words:
        s = rng.choice(SENTENCES)
        out.append(s)
        count += len(s.split())
    return " ".join(out)


def lognormal_int(rng, median, sigma, lo, hi) -> int:
    """Sample a skewed (log-normal) length, clipped to [lo, hi]."""
    return int(min(hi, max(lo, rng.lognormvariate(math.log(median), sigma))))


def chat_workload(rng, n):
    """Chat requests with log-normal input and output lengths."""
    rows = []
    for i in range(n):
        context_words = lognormal_int(rng, median=150, sigma=1.0, lo=10, hi=1500)
        max_tokens = lognormal_int(rng, median=200, sigma=0.7, lo=32, hi=1024)
        prompt = f"{make_text(rng, context_words)}\n\n{rng.choice(QUESTIONS)}"
        rows.append({
            "id": f"chat-{i:04d}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        })
    return rows


def shared_prefix_workload(rng, n):
    """Requests sharing one ~2k-token system prompt, each with a short question."""
    system = "You are a helpful analyst. Use this background:\n" + make_text(rng, 1500)
    rows = []
    for i in range(n):
        question = f"{make_text(rng, rng.randint(10, 40))} {rng.choice(QUESTIONS)}"
        rows.append({
            "id": f"prefix-{i:04d}",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "max_tokens": 128,
        })
    return rows


def long_context_workload(rng, n):
    """~8k-token documents to summarize, short outputs."""
    rows = []
    for i in range(n):
        doc = make_text(rng, 6000)  # ≈ 8k tokens
        rows.append({
            "id": f"long-{i:04d}",
            "messages": [{"role": "user", "content": f"{doc}\n\nSummarize this document."}],
            "max_tokens": 128,
        })
    return rows


def write_jsonl(path: Path, rows):
    """Write one JSON object per line."""
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):4d} requests -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="bench/workloads")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-chat", type=int, default=500)
    p.add_argument("--num-prefix", type=int, default=200)
    p.add_argument("--num-long", type=int, default=50)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Separate RNG per workload, so changing one count never changes the other files.
    write_jsonl(out / "chat.jsonl", chat_workload(random.Random(args.seed), args.num_chat))
    write_jsonl(out / "shared_prefix.jsonl",
                shared_prefix_workload(random.Random(args.seed + 1), args.num_prefix))
    write_jsonl(out / "long_context.jsonl",
                long_context_workload(random.Random(args.seed + 2), args.num_long))


if __name__ == "__main__":
    main()

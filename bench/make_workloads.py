"""
TokenForge — workload generator.

WHY THIS EXISTS: A benchmark is only meaningful if every run sends the SAME
requests. So we generate the workloads once, save them as files, commit them
to git, and never change them. If you change a workload, every old result
becomes incomparable — so make a new file (chat_v2.jsonl) instead.

Each workload stresses a different part of the system:

  chat.jsonl          Mixed lengths, like real chat traffic. The general case.
  shared_prefix.jsonl Every request starts with the same long system prompt.
                      Shows off prefix caching (Phase 2) and cache-aware
                      routing (Phase 6): the shared part can be computed once.
  long_context.jsonl  Very long inputs, short outputs. Stresses PREFILL
                      (compute-bound) and KV cache memory.

The text is synthetic (no internet needed, and fully reproducible from the
seed). What matters for performance is the LENGTH distribution, not whether
the prompts are interesting. Lengths here are in words; a word is ~1.3
tokens for English. The benchmark records exact token counts from the server.

Run:
    python bench/make_workloads.py --out-dir bench/workloads
"""

import argparse
import json
import math
import random
from pathlib import Path

# A bank of neutral sentences we shuffle into "documents" of any length.
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
    """
    Real request lengths are skewed: most are short, a few are very long.
    A log-normal distribution has exactly that shape. We clip the extremes.
    """
    return int(min(hi, max(lo, rng.lognormvariate(math.log(median), sigma))))


def chat_workload(rng, n):
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
    # ONE system prompt (~1,500 words ≈ 2,000 tokens) reused by every request.
    # Real apps do this constantly: long instructions + a short user question.
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
    # JSONL = one JSON object per line. Easy to stream, diff and append to.
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows):4d} requests -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="bench/workloads")
    p.add_argument("--seed", type=int, default=42)  # same seed = same files
    p.add_argument("--num-chat", type=int, default=500)
    p.add_argument("--num-prefix", type=int, default=200)
    p.add_argument("--num-long", type=int, default=50)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # A separate RNG per workload, so changing one count never changes the
    # contents of the other files.
    write_jsonl(out / "chat.jsonl", chat_workload(random.Random(args.seed), args.num_chat))
    write_jsonl(out / "shared_prefix.jsonl",
                shared_prefix_workload(random.Random(args.seed + 1), args.num_prefix))
    write_jsonl(out / "long_context.jsonl",
                long_context_workload(random.Random(args.seed + 2), args.num_long))


if __name__ == "__main__":
    main()
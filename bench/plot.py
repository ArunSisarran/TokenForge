"""
TokenForge — plot benchmark results.

Draws the chart you will put in your README: throughput (x) vs latency (y),
one line per run, one point per concurrency level.

WHY THIS CHART: serving is a trade-off. Pushing more concurrent requests
raises throughput (cheaper per token) but also raises latency (worse for each
user). Every engine/config traces a curve. A better system's curve sits
further RIGHT (more throughput) and LOWER (less latency). A single number
like "500 tok/s" hides the latency you paid for it; the curve doesn't.

Run:
    python bench/plot.py bench/results/*.json --out bench/results/latency_vs_throughput.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to a file; no display needed (works on servers)
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+", help="result JSON files from loadgen.py")
    p.add_argument("--out", default="bench/results/latency_vs_throughput.png")
    args = p.parse_args()

    fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(12, 4.5))

    for path in args.results:
        data = json.loads(Path(path).read_text())
        label = data["metadata"]["args"]["label"]
        rows = data["summary"]
        x = [r["output_tok_per_s"] for r in rows]
        # p95, not p50: we care about the unlucky users too.
        ttft = [r["ttft_ms_p95"] for r in rows]
        tpot = [r["tpot_ms_p95"] for r in rows]

        for ax, y in ((ax_ttft, ttft), (ax_tpot, tpot)):
            ax.plot(x, y, marker="o", label=label)
            for xi, yi, r in zip(x, y, rows):
                if yi is not None:
                    ax.annotate(f"c={r['concurrency']}", (xi, yi),
                                textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax_ttft.set_title("Time to first token (p95) vs throughput")
    ax_ttft.set_ylabel("TTFT p95 (ms)")
    ax_tpot.set_title("Time per output token (p95) vs throughput")
    ax_tpot.set_ylabel("TPOT p95 (ms)")
    for ax in (ax_ttft, ax_tpot):
        ax.set_xlabel("Output throughput (tokens/s)")
        ax.set_yscale("log")  # latencies span orders of magnitude across configs
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
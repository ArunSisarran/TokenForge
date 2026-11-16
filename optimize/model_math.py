"""
TokenForge — Phase 1: back-of-envelope model math.

Predict before you measure. Reads a model's config and computes, for
FP16 / FP8 / INT4 weights:
  - GPU memory used by the weights
  - bytes read from memory to generate ONE token
  - the fastest possible decode speed (the memory-bandwidth ceiling)
  - how many tokens of KV cache fit in the memory left over

No GPU needed: only the small config.json file is downloaded.

Run:
    python optimize/model_math.py --model Qwen/Qwen3-8B
"""

import argparse

from transformers import AutoConfig

# Bytes per weight for each precision we will test.
WEIGHT_BYTES = {"fp16": 2.0, "fp8": 1.0, "int4": 0.5}


def count_params(cfg) -> dict:
    h = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or h // n_heads
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size

    attn = (h * n_heads * head_dim       # Q projection
            + 2 * h * n_kv * head_dim    # K and V projections
            + n_heads * head_dim * h)    # output projection
    mlp = 3 * h * inter                  # gate, up and down projections
    embed = vocab * h
    lm_head = 0 if cfg.tie_word_embeddings else vocab * h

    return {
        "blocks": n_layers * (attn + mlp),
        "embed": embed,
        "lm_head": lm_head,
        "n_layers": n_layers,
        "n_kv": n_kv,
        "head_dim": head_dim,
    }

def kv_bytes_per_token(p: dict, kv_bytes: float = 2.0) -> float:
    # Every layer stores one Key and one Value vector per KV head, per token.
    return 2 * p["n_layers"] * p["n_kv"] * p["head_dim"] * kv_bytes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--bandwidth-gbs", type=float, default=300.0,
                    help="GPU memory bandwidth in GB/s (NVIDIA L4: 300)")
    ap.add_argument("--gpu-mem-mib", type=float, default=23034,
                    help="total GPU memory as nvidia-smi reports it")
    ap.add_argument("--mem-util", type=float, default=0.90,
                    help="fraction of GPU memory the server may use")
    ap.add_argument("--context", type=int, nargs="+", default=[512, 8192],
                    help="context lengths (tokens) to predict decode speed at")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    p = count_params(cfg)
    total = p["blocks"] + p["embed"] + p["lm_head"]
    # A model with tied embeddings reuses the embedding matrix as its lm_head.
    head = p["lm_head"] or p["embed"]
    kv_tok = kv_bytes_per_token(p)
    usable = args.gpu_mem_mib * 2**20 * args.mem_util
    bw = args.bandwidth_gbs * 1e9

    print(f"{args.model}")
    print(f"  parameters: {total / 1e9:.2f}B "
          f"(transformer blocks {p['blocks'] / 1e9:.2f}B, "
          f"embeddings {p['embed'] / 1e9:.2f}B, lm_head {p['lm_head'] / 1e9:.2f}B)")
    print(f"  KV cache per token (16-bit): {kv_tok / 1024:.0f} KB")
    print(f"  GPU: {args.bandwidth_gbs:.0f} GB/s, "
          f"{usable / 1e9:.1f} GB usable at {args.mem_util:.0%}\n")

    header = f"{'weights':>8} {'memory':>9} {'KV room':>12}"
    for t in args.context:
        header += f" | ctx {t:>5}: {'read/tok':>9} {'TPOT min':>9} {'tok/s max':>9}"
    print(header)
    print("-" * len(header))

    for name, wb in WEIGHT_BYTES.items():
        # Only the transformer blocks get quantized. Embeddings and lm_head
        # stay 16-bit, which is what typical quantization recipes do.
        resident = p["blocks"] * wb + (p["embed"] + p["lm_head"]) * 2
        kv_room = max(0.0, usable - resident) / kv_tok
        row = f"{name:>8} {resident / 1e9:>7.2f}GB {kv_room / 1e3:>8.0f}k tok"
        for t in args.context:
            # Bytes read to produce ONE token at batch size 1:
            #   all block weights + the lm_head + the KV cache of every earlier token.
            # The embedding table is NOT read in full: decoding looks up one row.
            per_step = p["blocks"] * wb + head * 2 + t * kv_tok
            tpot_ms = per_step / bw * 1000
            row += (f" |            {per_step / 1e9:>7.2f}GB"
                    f" {tpot_ms:>7.1f}ms {1000 / tpot_ms:>9.1f}")
        print(row)


if __name__ == "__main__":
    main()
"""
TokenForge Phase 1: quantize the model to FP8 (W8A8 dynamic) or INT4 (GPTQ
W4A16, group 128) with LLM Compressor, keeping lm_head 16-bit.

Run from the repo root, on a GPU (needs ~20 GB of free CPU RAM):
    python -m optimize.quantize --scheme fp8
    python -m optimize.quantize --scheme int4
"""

import argparse
import resource
import time
from pathlib import Path

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

# scheme -> (recipe, output folder suffix). W4A16 is symmetric int4, group size 128.
SCHEMES = {
    "fp8": (QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC",
                                 ignore=["lm_head"]), "FP8-Dynamic"),
    "int4": (GPTQModifier(targets="Linear", scheme="W4A16",
                          ignore=["lm_head"]), "W4A16-G128"),
}


def calibration_set(tokenizer, n: int, max_len: int, seed: int):
    """Tokenize n random ultrachat conversations, formatted with the chat template."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed).select(range(n))

    def tokenize(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        # The template already wrote the special tokens.
        return tokenizer(text, max_length=max_len, truncation=True,
                         add_special_tokens=False)

    return ds.map(tokenize, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=SCHEMES)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out-dir", default=None,
                    help="default: /workspace/models/<model name>-<scheme suffix>")
    ap.add_argument("--num-samples", type=int, default=512,
                    help="calibration samples (int4 only)")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Without a GPU, GPTQ silently runs on the CPU for hours.
    assert torch.cuda.is_available(), "quantization needs a GPU"
    recipe, suffix = SCHEMES[args.scheme]
    out_dir = Path(args.out_dir or
                   f"/workspace/models/{args.model.split('/')[-1]}-{suffix}")

    # Loaded on the CPU; LLM Compressor moves one layer at a time to the GPU.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")

    t0 = time.perf_counter()
    if args.scheme == "int4":
        ds = calibration_set(tokenizer, args.num_samples, args.max_seq_len, args.seed)
        oneshot(model=model, recipe=recipe, dataset=ds,
                max_seq_length=args.max_seq_len,
                num_calibration_samples=args.num_samples)
    else:
        oneshot(model=model, recipe=recipe)
    t_quant = time.perf_counter() - t0

    # Sanity check: broken quantization usually shows up as gibberish.
    dispatch_model(model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain a KV cache in one sentence."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    print("\nSample:", tokenizer.decode(out[0, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True))

    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    size = sum(f.stat().st_size for f in out_dir.glob("*.safetensors"))
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB on Linux
    print(f"\nScheme   {args.scheme} -> {out_dir}")
    print(f"Time     {t_quant / 60:.1f} min to quantize")
    print(f"On disk  {size / 1e9:.2f} GB of weights")
    print(f"Peak RAM {peak_ram:.1f} GB (CPU)")


if __name__ == "__main__":
    main()

"""
Vocabulary Pruning for MoE Whisper Models

Prunes a MoE (Mixture-of-Experts) Whisper model's vocabulary by directly
manipulating safetensors weights. Preserves the MoE architecture (moe_layers,
experts, router, shared_expert). Does not use WhisperForConditionalGeneration.

Token selection strategy:
  - Keeps all special tokens (e.g. <|startoftranscript|>, <|endoftext|>)
  - Keeps tokens appearing in the corpus (English-only, Devanagari stripped)
  - Keeps a curated list of must-keep tokens (fillers, STEM terms, common Hinglish)
  - Keeps short tokens (<=2 chars, or Ġ-prefixed <=3 chars) that are not Devanagari
  - Removes Devanagari-related tokens (byte-pair encoded or Unicode range)

Output: Pruned model in FP16 with updated config, tokenizer, and generation_config.

Usage:
    python prune_vocab.py \\
        --model_path ./checkpoint-102000 \\
        --tokenizer_path ./checkpoint-102000 \\
        --corpus_csv ./transcripts.csv \\
        --output_dir ./pruned_model \\
        [--text_column text] \\
        [--max_lines 500000]
"""

import argparse
import csv
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import WhisperTokenizer

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
DEVANAGARI_BYTES = re.compile(r"à¤|à¥|Ġà¤|Ġà¥")

MUST_KEEP_TOKENS = [
    "uh", "um", "ah", "oh", "hmm", "hm",
    "Ġuh", "Ġum", "Ġah", "Ġoh", "Ġhmm",
    "theta", "alpha", "beta", "gamma", "delta", "epsilon", "sigma", "omega", "pi",
    "Ġtheta", "Ġalpha", "Ġbeta", "Ġgamma", "Ġdelta", "Ġepsilon", "Ġsigma", "Ġomega", "Ġpi",
    "sin", "cos", "tan", "log", "sqrt", "plus", "minus", "by",
    "Ġsin", "Ġcos", "Ġtan", "Ġlog", "Ġsqrt", "Ġplus", "Ġminus", "Ġby",
    "mein", "yeh", "toh", "kyu", "kyun", "hai", "hain", "nahi",
    "Ġmein", "Ġyeh", "Ġtoh", "Ġkyu", "Ġkyun", "Ġhai", "Ġhain", "Ġnahi",
]


def is_devanagari_byte_token(token: str) -> bool:
    """Check if token contains Devanagari byte-pair encoding or Unicode."""
    if DEVANAGARI_BYTES.search(token):
        return True
    decoded = token.replace("Ġ", " ").replace("Ċ", "\n")
    return bool(DEVANAGARI_PATTERN.search(decoded))


def load_corpus_tokens(
    tokenizer: WhisperTokenizer,
    corpus_csv: str,
    text_column: str,
    max_lines: int = 500000,
) -> set:
    """Load unique token IDs from corpus CSV (English text only, Devanagari stripped)."""
    token_ids = set()
    if not os.path.exists(corpus_csv):
        print(f"  WARNING: Corpus not found at {corpus_csv}")
        return token_ids
    print(f"  Loading corpus: {corpus_csv}")
    count = 0
    with open(corpus_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        col = text_column if text_column in cols else cols[-1] if cols else "text"
        for row in reader:
            text = row.get(col, "")
            if not text:
                continue
            english_text = DEVANAGARI_PATTERN.sub("", text)
            if english_text.strip():
                ids = tokenizer.encode(english_text, add_special_tokens=False)
                token_ids.update(ids)
            count += 1
            if count >= max_lines:
                break
            if count % 100000 == 0:
                print(f"    {count} lines, {len(token_ids)} tokens")
    print(f"  Corpus: {count} lines -> {len(token_ids)} unique token IDs")
    return token_ids


def compute_kept_tokens(tokenizer: WhisperTokenizer, corpus_tokens: set) -> list[int]:
    """Compute which token IDs to keep based on corpus, Devanagari removal, and must-keep list."""
    vocab = tokenizer.get_vocab()
    devanagari_tokens = set()
    for token, idx in vocab.items():
        if is_devanagari_byte_token(token):
            devanagari_tokens.add(idx)

    kept_tokens = set()

    for token, idx in vocab.items():
        if token.startswith("<|") and token.endswith("|>"):
            kept_tokens.add(idx)

    for token in MUST_KEEP_TOKENS:
        if token in vocab:
            kept_tokens.add(vocab[token])

    english_corpus = corpus_tokens - devanagari_tokens
    kept_tokens.update(english_corpus)

    for token, idx in vocab.items():
        if idx not in devanagari_tokens:
            if len(token) <= 2 or (token.startswith("Ġ") and len(token) <= 3):
                if not is_devanagari_byte_token(token):
                    kept_tokens.add(idx)

    kept_tokens = kept_tokens - devanagari_tokens
    return sorted(list(kept_tokens))


def prune_model(
    model_path: str,
    tokenizer_path: str,
    output_dir: str,
    kept_ids: list[int],
) -> dict:
    """Prune model vocab by slicing embed_tokens and proj_out, update config and tokenizer."""
    print(f"\n{'='*60}")
    print("Pruning model")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    new_vocab_size = len(kept_ids)
    kept_tensor = torch.tensor(kept_ids, dtype=torch.long)

    new_state = {}
    total_params = 0
    active_params = 0

    for key, tensor in state_dict.items():
        if key == "model.decoder.embed_tokens.weight":
            new_tensor = tensor[kept_tensor].clone().half()
            new_state[key] = new_tensor
        elif key == "proj_out.weight":
            new_tensor = tensor[kept_tensor].clone().half()
            new_state[key] = new_tensor
        else:
            new_state[key] = tensor.half()

        n = new_state[key].numel()
        total_params += n
        if "experts" in key and "shared" not in key:
            active_params += n // 2
        else:
            active_params += n

    save_file(new_state, os.path.join(output_dir, "model.safetensors"))

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
    original_vocab_size = state_dict["model.decoder.embed_tokens.weight"].shape[0]

    # Copy and fix config
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["vocab_size"] = new_vocab_size
    for key in ["bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"]:
        if key in cfg and cfg[key] in old_to_new:
            cfg[key] = old_to_new[cfg[key]]
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Copy and fix generation_config
    gen_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            gc = json.load(f)
        gc["vocab_size"] = new_vocab_size
        for key in ["decoder_start_token_id", "eos_token_id", "pad_token_id", "no_timestamps_token_id"]:
            if key in gc and gc[key] in old_to_new:
                gc[key] = old_to_new[gc[key]]
        if "suppress_tokens" in gc and gc["suppress_tokens"]:
            gc["suppress_tokens"] = [old_to_new[t] for t in gc["suppress_tokens"] if t in old_to_new]
        if "begin_suppress_tokens" in gc and gc["begin_suppress_tokens"]:
            gc["begin_suppress_tokens"] = [old_to_new[t] for t in gc["begin_suppress_tokens"] if t in old_to_new]
        with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
            json.dump(gc, f, indent=2)

    # Copy and fix tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path)
    vocab = tokenizer.get_vocab()
    new_vocab = {tok: old_to_new[oid] for tok, oid in vocab.items() if oid in old_to_new}
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(new_vocab, f, ensure_ascii=False)

    for fname in [
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "normalizer.json",
    ]:
        for src_dir in [tokenizer_path, model_path]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, output_dir)
                break

    # Fix tokenizer_config
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        if "added_tokens_decoder" in tc:
            new_dec = {}
            for oid_str, info in tc["added_tokens_decoder"].items():
                oid = int(oid_str)
                if oid in old_to_new:
                    new_dec[str(old_to_new[oid])] = info
            tc["added_tokens_decoder"] = new_dec
        with open(tc_path, "w") as f:
            json.dump(tc, f, indent=2)

    at_path = os.path.join(output_dir, "added_tokens.json")
    if os.path.exists(at_path):
        with open(at_path) as f:
            at = json.load(f)
        new_at = {tok: old_to_new[oid] for tok, oid in at.items() if oid in old_to_new}
        with open(at_path, "w") as f:
            json.dump(new_at, f, indent=2)

    size_mb = os.path.getsize(os.path.join(output_dir, "model.safetensors")) / (1024**2)
    orig_size_mb = os.path.getsize(os.path.join(model_path, "model.safetensors")) / (1024**2)

    print(f"  Vocab: {original_vocab_size} -> {new_vocab_size}")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Active params: {active_params:,} ({active_params/1e6:.2f}M)")
    print(f"  Original size: {orig_size_mb:.1f} MB (FP32)")
    print(f"  Pruned size:   {size_mb:.1f} MB (FP16)")
    print(f"  Saved to: {output_dir}")

    saved = load_file(os.path.join(output_dir, "model.safetensors"))
    moe_keys = [k for k in saved.keys() if "moe" in k or "expert" in k or "router" in k]
    print(f"  MoE keys preserved: {len(moe_keys)}")

    return {
        "total_params": total_params,
        "active_params": active_params,
        "vocab": new_vocab_size,
        "size_mb": size_mb,
        "path": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune MoE Whisper model vocabulary by corpus-based token filtering."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: same as model_path)",
    )
    parser.add_argument(
        "--corpus_csv",
        type=str,
        required=True,
        help="Path to CSV with transcript text for corpus-based token selection",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to write pruned model",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="CSV column name for transcript text (default: text)",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=500000,
        help="Max corpus lines to process (default: 500000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_path

    print("Loading tokenizer for corpus processing...")
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path)
    corpus_tokens = load_corpus_tokens(
        tokenizer, args.corpus_csv, args.text_column, args.max_lines
    )
    kept_ids = compute_kept_tokens(tokenizer, corpus_tokens)
    print(f"Kept tokens: {len(kept_ids)} (from {len(tokenizer.get_vocab())})")

    prune_model(args.model_path, tokenizer_path, args.output_dir, kept_ids)


if __name__ == "__main__":
    main()

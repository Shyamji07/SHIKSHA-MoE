"""
SHIKSHA-MoE Evaluation Script

Evaluates a trained SHIKSHA-MoE checkpoint on a CSV test set.
Computes WER, CER, STEM-WER, and per-utterance latency.

Usage:
    python eval_model.py \
        --checkpoint_dir ./output/shiksha-moe-4E4D/checkpoint-45000 \
        --test_csv ./dataset/stem_hinglish_test.csv \
        --output_dir ./results \
        --stem_glossary ./dataset/stem_glossary.txt
"""

import os
import re
import gc
import time
import string
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import jiwer
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from safetensors.torch import load_file

from model import (
    HybridMoEEncoderLayer,
    HybridMoSEEncoderWrapper,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SHIKSHA-MoE on a test CSV"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--processor", type=str, default="openai/whisper-tiny",
        help="Whisper processor path",
    )
    parser.add_argument("--stem_glossary", type=str, default=None,
                        help="Path to STEM glossary (one term per line)")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--expert_dim", type=int, default=256)
    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    return parser.parse_args()


def normalize_text(text):
    """Normalize text for WER/CER computation."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    mappings = {"kyu": "kyun", "kyoon": "kyun", "hai": "hain", "hein": "hain"}
    words = text.split()
    text = " ".join([mappings.get(w, w) for w in words])
    return re.sub(r"\s+", " ", text).strip()


def compute_stem_wer(reference, hypothesis, glossary):
    """
    Compute STEM-WER: WER restricted to STEM glossary terms.

    STEM-WER = (S_T + D_T) / N_T
    where N_T = count of reference tokens in glossary,
    S_T = substituted STEM tokens, D_T = deleted STEM tokens.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    ref_stem = [w for w in ref_words if w in glossary]
    if not ref_stem:
        return None

    alignment = jiwer.process_words(
        " ".join(ref_words), " ".join(hyp_words)
    )

    stem_errors = 0
    for chunk in alignment.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if idx < len(ref_words) and ref_words[idx] in glossary:
                    stem_errors += 1

    return stem_errors / len(ref_stem)


def load_moe_model(checkpoint_dir, args, device):
    """Load a SHIKSHA-MoE model from a training checkpoint."""
    config = WhisperConfig.from_pretrained(checkpoint_dir)
    config.encoder_layers = args.num_encoder_layers
    config.decoder_layers = args.num_decoder_layers

    model = WhisperForConditionalGeneration(config)
    moe_encoder_layers = nn.ModuleList([
        HybridMoEEncoderLayer(
            config, args.num_experts, args.top_k,
            args.expert_dim, args.shared_dim,
        )
        for _ in range(args.num_encoder_layers)
    ])
    moe_encoder = HybridMoSEEncoderWrapper(
        model.model.encoder, moe_encoder_layers
    )
    model.model.encoder = moe_encoder

    ckpt_file = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(ckpt_file):
        state_dict = load_file(ckpt_file)
    else:
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location="cpu",
        )
    model.load_state_dict(state_dict, strict=False)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = None
        model.generation_config.begin_suppress_tokens = None

    return model.to(device).eval()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    glossary = set()
    if args.stem_glossary and os.path.exists(args.stem_glossary):
        with open(args.stem_glossary) as f:
            glossary = {line.strip().lower() for line in f if line.strip()}
        print(f"Loaded STEM glossary: {len(glossary)} terms")

    processor = WhisperProcessor.from_pretrained(args.processor)
    df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(df)}")

    model = load_moe_model(args.checkpoint_dir, args, device)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        audio_path = row["audio_path"]
        gt = row["ground_truth"]
        if not os.path.exists(audio_path):
            continue
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception:
            continue

        with torch.no_grad():
            feats = processor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).input_features.to(device)
            t0 = time.time()
            pred_ids = model.generate(
                feats, max_new_tokens=225, use_cache=True
            )
            latency = time.time() - t0
            transcript = processor.batch_decode(
                pred_ids, skip_special_tokens=True
            )[0]

        gt_n = normalize_text(gt)
        pred_n = normalize_text(transcript)
        if not gt_n and not pred_n:
            wer_val, cer_val = 0.0, 0.0
        elif not gt_n or not pred_n:
            wer_val, cer_val = 1.0, 1.0
        else:
            wer_val = jiwer.wer(gt_n, pred_n)
            cer_val = jiwer.cer(gt_n, pred_n)

        entry = {
            "audio_path": audio_path,
            "ground_truth": gt,
            "prediction": transcript,
            "wer": wer_val,
            "cer": cer_val,
            "latency_s": latency,
        }
        if glossary:
            sw = compute_stem_wer(gt, transcript, glossary)
            entry["stem_wer"] = sw

        results.append(entry)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "evaluation_detailed.csv")
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated: {len(results_df)}")
    print(f"WER:  {results_df['wer'].mean() * 100:.2f}%")
    print(f"CER:  {results_df['cer'].mean() * 100:.2f}%")
    if glossary and "stem_wer" in results_df.columns:
        valid = results_df["stem_wer"].dropna()
        print(f"STEM-WER: {valid.mean() * 100:.2f}% "
              f"({len(valid)} utterances with STEM terms)")
    print(f"Avg latency: {results_df['latency_s'].mean() * 1000:.1f} ms")
    print(f"Results saved to: {out_path}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

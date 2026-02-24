"""
SHIKSHA-MoE Inference Script

Transcribe a single audio file or a directory of audio files.

Usage:
    # Single file
    python inference.py \
        --checkpoint_dir ./output/shiksha-moe-4E4D/checkpoint-45000 \
        --audio path/to/audio.wav

    # Directory
    python inference.py \
        --checkpoint_dir ./output/shiksha-moe-4E4D/checkpoint-45000 \
        --audio_dir path/to/audios/ \
        --output_csv transcriptions.csv
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import librosa
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperConfig, WhisperForConditionalGeneration
from safetensors.torch import load_file

from model import HybridMoEEncoderLayer, HybridMoSEEncoderWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="SHIKSHA-MoE inference")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to a single audio file")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory of audio files to transcribe")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--processor", type=str, default="openai/whisper-tiny")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--expert_dim", type=int, default=256)
    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    return parser.parse_args()


def load_model(checkpoint_dir, args, device):
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


def transcribe(model, processor, audio_path, device):
    audio, sr = librosa.load(audio_path, sr=16000)
    feats = processor(
        audio, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        t0 = time.time()
        pred_ids = model.generate(feats, max_new_tokens=225, use_cache=True)
        latency = time.time() - t0

    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return text, latency


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = WhisperProcessor.from_pretrained(args.processor)
    model = load_model(args.checkpoint_dir, args, device)
    print("Model loaded.")

    if args.audio:
        text, latency = transcribe(model, processor, args.audio, device)
        print(f"\nTranscription: {text}")
        print(f"Latency: {latency * 1000:.1f} ms")

    elif args.audio_dir:
        import pandas as pd
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        files = sorted([
            os.path.join(args.audio_dir, f)
            for f in os.listdir(args.audio_dir)
            if os.path.splitext(f)[1].lower() in audio_exts
        ])
        print(f"Found {len(files)} audio files")

        results = []
        for fpath in tqdm(files, desc="Transcribing"):
            try:
                text, latency = transcribe(model, processor, fpath, device)
                results.append({
                    "audio_path": fpath,
                    "transcription": text,
                    "latency_s": latency,
                })
            except Exception as e:
                print(f"  Error on {fpath}: {e}")

        df = pd.DataFrame(results)
        out = args.output_csv or "transcriptions.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} transcriptions to {out}")

    else:
        print("Provide --audio or --audio_dir")


if __name__ == "__main__":
    main()

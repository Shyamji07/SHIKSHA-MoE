# SHIKSHA-MoE

**Shared Hybrid Inference for Knowledge in STEM Hinglish ASR**

> Preventing Semantic Feature Dissociation in Technical Hinglish ASR via Shared-Expert Sparsity

This repository contains the code and evaluation benchmark for the paper:

**"SHIKSHA-MoE: Preventing Semantic Feature Dissociation in Technical Hinglish ASR via Shared-Expert Sparsity"** (Interspeech 2026)

---

## Overview

SHIKSHA-MoE is a Mixture-of-Experts (MoE) architecture for automatic speech recognition (ASR) that addresses **Semantic Feature Dissociation (SFD)** — a failure mode where MoE-based encoder sparsification scatters co-adapted neurons across experts, producing repetition loops and meaning-altering hallucinations on technical terminology.

### Key Idea

When compressing a pre-trained Whisper encoder via MoE partitioning, neurons that jointly encode domain terms (e.g., "bryophytes") are scattered across experts. SHIKSHA-MoE prevents this by introducing a **shared always-active expert** that anchors global linguistic structure, while sparse routed experts specialize for domain features.

### Architecture

```
Input x
  ├── Shared Expert (256 neurons, always active) ──┐
  └── Router → Top-2 of 4 Routed Experts (256 each) ──┤
                                                       ⊕ → Output
```

- **Active neurons per token**: 256 (shared) + 2 × 256 (routed) = 768
- **Dense baseline**: 1536 neurons → **50% encoder FFN FLOP reduction**
- **Memory footprint**: Virtually unchanged (37M → 35M active, all 37M stored)

### Results (STEM-Hinglish-1.2K)

| Model | Params | WER ↓ | STEM-WER ↓ | Semantic Reliability |
|-------|--------|-------|------------|---------------------|
| Teacher (Whisper Small) | 242M | 6.0% | — | — |
| Dense (4E-4D) | 37M | 9.5% | 14.2% | 96% |
| Partitioned MoE | 33M | 9.9% | 16.8% | 71% |
| **SHIKSHA-MoE (4E-4D)** | **37M** | **8.3%** | **9.8%** | **97%** |

---

## Repository Structure

```
SHIKSHA-MoE/
├── code/
│   ├── model.py          # SHIKSHA-MoE architecture (all MoE components)
│   ├── train.py          # Training with knowledge distillation
│   ├── eval_model.py     # Evaluation (WER, CER, STEM-WER, latency)
│   ├── inference.py      # Single-file and batch transcription
│   ├── requirements.txt  # Python dependencies
│   └── README.md         # This file
└── dataset/
    ├── stem_hinglish_test.csv   # 3,500 test utterances with ground truth
    ├── stem_glossary.txt        # 2,400 STEM technical terms
    └── audios/                  # Audio files (WAV, 16kHz)
```

---

## Installation

```bash
git clone https://github.com/Allen-Career-Institute/SHIKSHA-MoE.git
cd SHIKSHA-MoE/code
pip install -r requirements.txt
```

---

## Usage

### 1. Training

Train SHIKSHA-MoE via knowledge distillation from a Whisper teacher:

```bash
python train.py \
    --base_model openai/whisper-tiny \
    --teacher_model openai/whisper-small \
    --data_dir /path/to/processed_chunks \
    --output_dir ./output/shiksha-moe-4E4D \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --num_experts 4 \
    --top_k 2 \
    --expert_dim 256 \
    --shared_dim 256 \
    --max_steps 45000
```

**Training details:**
- Loss: Cross-Entropy + KL Divergence (temperature=2.0, λ=0.5) + Load Balancing (0.01)
- Optimizer: AdamW, lr=1e-5, cosine schedule, 2000 warmup steps
- Hardware: NVIDIA A100, FP16
- Data: 1,200 hours of Hindi-English STEM educational lectures

### 2. Evaluation

Evaluate on the released test set with STEM-WER:

```bash
python eval_model.py \
    --checkpoint_dir ./output/shiksha-moe-4E4D/checkpoint-45000 \
    --test_csv ../dataset/stem_hinglish_test.csv \
    --stem_glossary ../dataset/stem_glossary.txt \
    --output_dir ./results
```

### 3. Inference

Transcribe a single audio file:

```bash
python inference.py \
    --checkpoint_dir ./output/shiksha-moe-4E4D/checkpoint-45000 \
    --audio path/to/audio.wav
```

Batch transcription:

```bash
python inference.py \
    --checkpoint_dir ./output/shiksha-moe-4E4D/checkpoint-45000 \
    --audio_dir path/to/audios/ \
    --output_csv transcriptions.csv
```

---

## Dataset: STEM-Hinglish-1.2K

We release a benchmark test set for evaluating code-switched STEM ASR:

| Characteristic | Value |
|----------------|-------|
| Test utterances | 3,500 |
| Duration | ~42 hours |
| Domains | Physics, Chemistry, Biology, Mathematics |
| Unique STEM terms | 2,400 |
| Code-switch points/utt | 2.4 (avg) |
| Transcription | 100% manually verified by bilingual STEM annotators |

**STEM-WER metric**: Evaluates recognition quality specifically on the 2,400 domain-critical technical terms, exposing errors like "bryophytes" → "bryophyll" that standard WER masks.

> **Note:** GitHub's web interface only displays the first 1,000 files in a directory. To access all 3,500 audio files, clone the repository:
> ```bash
> git clone https://github.com/Shyamji07/SHIKSHA-MoE.git
> ls SHIKSHA-MoE/dataset/audios/ | wc -l  # Output: 3500
> ```

### CSV Format

```csv
audio_path,ground_truth
audios/stem_hinglish_00042.wav,"sir is question mein zero point five per female intrinsic growth rate..."
```

---

## STEM-WER Metric

Standard WER treats "the" and "bryophytes" equally. STEM-WER isolates recognition quality on domain-critical terms:

```
STEM-WER = (S_T + D_T) / N_T
```

where:
- `N_T` = count of reference tokens in the STEM glossary
- `S_T` = substituted STEM tokens
- `D_T` = deleted STEM tokens

---

## Architecture Details

### Configurations

| Config | Encoder | Decoder | Params | Active Neurons |
|--------|---------|---------|--------|----------------|
| 4E-2D | 4 layers | 2 layers | 32M | 768/token |
| 4E-4D | 4 layers | 4 layers | 37M | 768/token |

### Weight Initialization

The shared expert is initialized from the **first 256 columns** of the pre-trained FFN weight matrix (most common linguistic patterns). Routed experts are initialized from subsequent columns, ensuring domain-specific neurons are distributed across specialists.

### Inference Efficiency

| Model | A100 GPU | CPU (Xeon) | CPU RTF |
|-------|----------|------------|---------|
| Dense (4E-4D) | 280ms | 680ms | 0.045 |
| SHIKSHA (4E-4D) | 265ms | 495ms | 0.033 |

27% CPU latency reduction (4E-4D); 33% (4E-2D).

---

## Citation

```bibtex
@inproceedings{shiksha_moe_2026,
    title={SHIKSHA-MoE: Preventing Semantic Feature Dissociation in Technical Hinglish ASR via Shared-Expert Sparsity},
    author={Anonymous},
    booktitle={Proc. Interspeech},
    year={2026},
    address={Pittsburgh, PA, USA},
}
```

---

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is released **for research purposes only**. Commercial use is strictly prohibited. See [LICENSE](../LICENSE) for details.

## Acknowledgments

This work builds on [OpenAI Whisper](https://github.com/openai/whisper) and the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.

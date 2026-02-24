"""
SHIKSHA-MoE Training Script

Trains SHIKSHA-MoE via knowledge distillation from a Whisper teacher model.
Loss = CrossEntropy + KL Divergence + Load Balancing

Usage:
    python train.py \
        --base_model openai/whisper-tiny \
        --teacher_model openai/whisper-small \
        --data_dir ./data/processed_chunks \
        --output_dir ./output/shiksha-moe-4E4D
"""

import os
import gc
import glob
import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

from model import create_shiksha_moe, compute_load_balancing_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SHIKSHA-MoE via knowledge distillation"
    )
    parser.add_argument(
        "--base_model", type=str, default="openai/whisper-tiny",
        help="Base Whisper model to initialize from",
    )
    parser.add_argument(
        "--teacher_model", type=str, default="openai/whisper-small",
        help="Teacher model for knowledge distillation",
    )
    parser.add_argument(
        "--processor", type=str, default="openai/whisper-tiny",
        help="Whisper processor/tokenizer path",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--expert_dim", type=int, default=256)
    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=45000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)

    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            inputs, return_tensors="pt"
        )
        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            labels, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        bos_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        if (labels[:, 0] == bos_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class ShikshaMoETrainer(Seq2SeqTrainer):
    """Custom trainer with KD loss and load-balancing loss."""

    def __init__(self, teacher_model, moe_encoder, temperature, kl_weight,
                 load_balance_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.moe_encoder = moe_encoder
        self.kl_loss_fn = nn.KLDivLoss(reduction="none")
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.load_balance_weight = load_balance_weight

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        outputs_student = model(**inputs)
        ce_loss = outputs_student.loss

        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs).logits

        p_student = F.log_softmax(
            outputs_student.logits / self.temperature, dim=-1
        )
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        min_len = min(p_student.shape[1], p_teacher.shape[1])

        kl = self.kl_loss_fn(
            p_student[:, :min_len, :], p_teacher[:, :min_len, :]
        ).sum(dim=-1)
        mask = (inputs["labels"][:, :min_len] >= 0).float()
        kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)
        kl_loss = kl_loss * (self.temperature ** 2)
        kl_loss = torch.clamp(kl_loss, max=5.0)

        load_balance_loss = torch.tensor(0.0, device=ce_loss.device)
        if (hasattr(self.moe_encoder, "router_probs_list")
                and self.moe_encoder.router_probs_list):
            for rp in self.moe_encoder.router_probs_list:
                load_balance_loss += compute_load_balancing_loss(rp)
            load_balance_loss /= len(self.moe_encoder.router_probs_list)

        loss = (ce_loss
                + self.kl_weight * kl_loss
                + self.load_balance_weight * load_balance_loss)
        return (loss, outputs_student) if return_outputs else loss


def load_dataset(data_dir):
    """
    Load HuggingFace dataset chunks from disk.

    Supports two layouts:
      1. Pre-split: data_dir contains train_chunk_* and optionally test_chunk_*
      2. Single dataset: data_dir is itself a saved Dataset (has dataset_info.json)
      3. Flat chunks: data_dir contains internal_chunk_* or chunk_*

    If no test split is found, 5% of training data is held out for evaluation.
    """
    single_ds_marker = os.path.join(data_dir, "dataset_info.json")
    if os.path.isfile(single_ds_marker):
        print(f"Loading single dataset from {data_dir}")
        full_ds = load_from_disk(data_dir)
        split = full_ds.train_test_split(test_size=0.05, seed=42)
        gc.collect()
        return DatasetDict({"train": split["train"], "test": split["test"]})

    train_chunks = sorted(
        glob.glob(os.path.join(data_dir, "train_chunk_*"))
    )
    test_chunks = sorted(
        glob.glob(os.path.join(data_dir, "test_chunk_*"))
    )

    if not train_chunks:
        train_chunks = sorted(
            glob.glob(os.path.join(data_dir, "internal_chunk_*"))
        )
    if not train_chunks:
        train_chunks = sorted(
            glob.glob(os.path.join(data_dir, "chunk_*"))
        )

    if not train_chunks:
        subdirs = sorted([
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and os.path.isfile(os.path.join(data_dir, d, "dataset_info.json"))
        ])
        if subdirs:
            train_chunks = subdirs
        else:
            raise ValueError(
                f"No dataset chunks found in {data_dir}. "
                f"Expected train_chunk_*, internal_chunk_*, chunk_* directories, "
                f"or a single dataset with dataset_info.json. "
                f"Contents: {os.listdir(data_dir)[:20]}"
            )

    print(f"Loading {len(train_chunks)} train chunks, "
          f"{len(test_chunks)} test chunks...")

    train_datasets = []
    for p in train_chunks:
        try:
            train_datasets.append(load_from_disk(p))
        except Exception as e:
            print(f"  Warning: skipping {p}: {e}")

    if not train_datasets:
        raise ValueError(
            f"All {len(train_chunks)} chunks failed to load from {data_dir}. "
            f"Verify the data was saved with datasets.Dataset.save_to_disk()."
        )

    train_ds = (train_datasets[0] if len(train_datasets) == 1
                else concatenate_datasets(train_datasets))

    if test_chunks:
        test_datasets = [load_from_disk(p) for p in test_chunks]
        test_ds = (test_datasets[0] if len(test_datasets) == 1
                   else concatenate_datasets(test_datasets))
    else:
        n_test = min(500, max(1, int(len(train_ds) * 0.05)))
        test_ds = train_ds.select(range(n_test))
        print(f"  No test chunks found; using first {n_test} train samples for eval")

    gc.collect()
    return DatasetDict({"train": train_ds, "test": test_ds})


def main():
    args = parse_args()

    print("=" * 60)
    print("SHIKSHA-MoE Training")
    print("=" * 60)
    print(f"Base model:    {args.base_model}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Architecture:  {args.num_encoder_layers}E-{args.num_decoder_layers}D")
    print(f"Experts:       1 shared ({args.shared_dim}) + "
          f"{args.num_experts} routed ({args.expert_dim}), Top-{args.top_k}")
    active = args.shared_dim + args.top_k * args.expert_dim
    print(f"Active neurons/token: {active}")

    model, moe_encoder = create_shiksha_moe(
        base_model_name=args.base_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        expert_dim=args.expert_dim,
        shared_dim=args.shared_dim,
    )

    print(f"Loading teacher: {args.teacher_model}")
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        args.teacher_model
    )
    teacher_model.eval()

    processor = WhisperProcessor.from_pretrained(args.processor)
    tokenizer = WhisperTokenizer.from_pretrained(args.processor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    teacher_model.to(device)

    dataset_dict = load_dataset(args.data_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=4,
        fp16=True,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=3000,
        save_steps=3000,
        logging_steps=100,
        save_total_limit=5,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=preds,
                                             references=refs)}

    trainer = ShikshaMoETrainer(
        teacher_model=teacher_model,
        moe_encoder=moe_encoder,
        temperature=args.temperature,
        kl_weight=args.kl_weight,
        load_balance_weight=args.load_balance_weight,
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    processor.save_pretrained(args.output_dir)

    print("\nStarting training...")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()

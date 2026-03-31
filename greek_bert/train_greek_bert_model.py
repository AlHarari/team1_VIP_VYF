#!/usr/bin/env python3

import os
import sys
import math
import argparse
import subprocess
from itertools import chain

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
from datasets import load_dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================
# Hardcoded controls
# =========================
NUM_GPUS = None   # None => use all visible GPUs; or set to 1, 2, 4, ...
USE_BF16 = False  # set True if your GPUs support bf16 well
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train_file",
        type=str,
        default="/storage/ice-shared/vip-vyf/embeddings_team/corpora/clean_corpora.bin",
    )
    p.add_argument(
        "--tokenizer_dir",
        type=str,
        default="/home/hice1/dharden7/scratch/greekbert_proj/tokenizer_grc_wordpiece_32895",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/hice1/dharden7/scratch/greekbert_proj/bert_output",
    )

    # LatinBERT defaults
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--global_batch_size", type=int, default=256)

    # practical controls
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    return p.parse_args()


def choose_num_gpus():
    visible = torch.cuda.device_count()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible CUDA devices: {visible}")

    if visible == 0:
        return 0

    if NUM_GPUS is None:
        chosen = visible
    else:
        chosen = min(NUM_GPUS, visible)

    print(f"Using {chosen} GPU(s)")
    return chosen


def maybe_launch_distributed(num_gpus):
    if num_gpus <= 1:
        return
    if "LOCAL_RANK" in os.environ:
        return

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        sys.argv[0],
        *sys.argv[1:],
    ]
    subprocess.run(cmd, check=True)
    sys.exit(0)


def main():
    args = parse_args()
    num_gpus = choose_num_gpus()
    maybe_launch_distributed(num_gpus)

    set_seed(SEED)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_dir)

    raw = load_dataset("text", data_files={"train": args.train_file})["train"]
    raw = raw.filter(lambda ex: ex["text"] is not None and ex["text"].strip() != "")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing",
    )

    chunk_size = args.max_seq_length - 2
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    def group_texts(examples):
        all_ids = list(chain.from_iterable(examples["input_ids"]))
        total_length = (len(all_ids) // chunk_size) * chunk_size
        all_ids = all_ids[:total_length]

        input_ids = []
        special_tokens_mask = []

        for i in range(0, total_length, chunk_size):
            chunk = all_ids[i : i + chunk_size]
            ids = [cls_id] + chunk + [sep_id]
            input_ids.append(ids)
            special_tokens_mask.append([1] + [0] * len(chunk) + [1])

        return {
            "input_ids": input_ids,
            "special_tokens_mask": special_tokens_mask,
        }

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        num_proc=args.preprocessing_num_workers,
        desc=f"Packing into {args.max_seq_length}-token blocks",
    )

    if len(lm_dataset) == 0:
        raise ValueError("No training blocks were created. Check your corpus and max_seq_length.")

    # From scratch: config -> random initialization
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = max(
        1,
        math.ceil(args.global_batch_size / (args.per_device_train_batch_size * world_size)),
    )
    effective_batch_size = args.per_device_train_batch_size * world_size * grad_accum

    use_bf16 = USE_BF16
    use_fp16 = torch.cuda.is_available() and not use_bf16

    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if trainer.is_world_process_zero():
        print(f"Tokenizer dir: {args.tokenizer_dir}")
        print(f"Train file: {args.train_file}")
        print(f"Output dir: {args.output_dir}")
        print(f"World size: {world_size}")
        print(f"Per-device batch size: {args.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum}")
        print(f"Effective global batch size: {effective_batch_size}")
        print(f"Training blocks: {len(lm_dataset)}")
        print("Model initialization: from scratch (random weights)")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if trainer.is_world_process_zero():
        print("Saved model and tokenizer to:", args.output_dir)


if __name__ == "__main__":
    main()
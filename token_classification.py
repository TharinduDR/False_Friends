"""
Token Classification for False Friend Detection using HuggingFace Transformers.

Loads data from HuggingFace Hub: false-friends/en_es_token
Source and target sentences are concatenated into a single sequence:
  [CLS] source_tokens [SEP] target_tokens [SEP]

Notebook usage:

    from token_classification import train_model, predict, evaluate_model

    # Train
    trainer, tokenizer = train_model(
        model_name="xlm-roberta-base",
        output_dir="./ff_model",
        dataset_name="false-friends/en_es_token",
        epochs=10,
        batch_size=16,
        lr=5e-5,
    )

    # Predict
    results = predict(
        model_path="./ff_model",
        source="This is a sensible solution",
        target="Esta es una solución sensible",
    )

    # Evaluate
    evaluate_model(
        model_path="./ff_model",
        dataset_name="false-friends/en_es_token",
    )

CLI usage:

    python token_classification.py train --model_name xlm-roberta-base --output_dir ./ff_model
    python token_classification.py predict --model_path ./ff_model --source "..." --target "..."
    python token_classification.py evaluate --model_path ./ff_model
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "false-friends/en_es_token"
LABEL_LIST = ["O", "B-FF"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — concatenated source + target
# ──────────────────────────────────────────────────────────────────────────────
class FalseFriendPairDataset(TorchDataset):
    """Each example concatenates source and target words into one sequence:
        [CLS] src_w1 src_w2 ... [SEP] tgt_w1 tgt_w2 ... [SEP]
    Labels are aligned for both halves; special tokens get -100.
    """

    def __init__(self, src_words, src_labels, tgt_words, tgt_labels, tokenizer, max_length=512):
        assert len(src_words) == len(tgt_words)
        self.src_words = src_words
        self.src_labels = src_labels
        self.tgt_words = tgt_words
        self.tgt_labels = tgt_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_words)

    def __getitem__(self, idx):
        s_words = self.src_words[idx]
        s_labels = self.src_labels[idx]
        t_words = self.tgt_words[idx]
        t_labels = self.tgt_labels[idx]

        encoding = self.tokenizer(
            s_words,
            t_words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Combined label list: source labels then target labels
        combined_labels = list(s_labels) + list(t_labels)

        # Align subword tokens to word-level labels
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                if word_id < len(combined_labels):
                    label_ids.append(combined_labels[word_id] if isinstance(combined_labels[word_id], int) else LABEL2ID[combined_labels[word_id]])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_id = word_id

        encoding["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading from HuggingFace Hub
# ──────────────────────────────────────────────────────────────────────────────
def load_hf_split(dataset_name, split, token=None):
    """Load a split from HuggingFace Hub and extract parallel lists.
    Returns: src_words, src_labels, tgt_words, tgt_labels"""
    ds = load_dataset(dataset_name, split=split, token=token)
    src_words = [row["source_words"] for row in ds]
    src_labels = [row["source_labels"] for row in ds]
    tgt_words = [row["target_words"] for row in ds]
    tgt_labels = [row["target_labels"] for row in ds]
    return src_words, src_labels, tgt_words, tgt_labels


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, label_ids):
        seq_preds, seq_labels = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(ID2LABEL[l])
            seq_preds.append(ID2LABEL[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def detailed_report(predictions, label_ids, src_words_list, tgt_words_list):
    """Print seqeval report and per-side breakdown."""
    preds = np.argmax(predictions, axis=-1)

    all_true, all_pred = [], []
    src_true, src_pred = [], []
    tgt_true, tgt_pred = [], []

    for i, (pred_seq, label_seq) in enumerate(zip(preds, label_ids)):
        seq_preds, seq_labels = [], []
        n_src = len(src_words_list[i]) if i < len(src_words_list) else 0

        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(ID2LABEL[l])
            seq_preds.append(ID2LABEL[p])

        all_true.append(seq_labels)
        all_pred.append(seq_preds)

        if n_src <= len(seq_labels):
            src_true.append(seq_labels[:n_src])
            src_pred.append(seq_preds[:n_src])
            tgt_true.append(seq_labels[n_src:])
            tgt_pred.append(seq_preds[n_src:])

    print("=== Overall ===")
    print(classification_report(all_true, all_pred))

    if src_true:
        print("=== Source (English) side ===")
        print(f"  F1:        {f1_score(src_true, src_pred):.4f}")
        print(f"  Precision: {precision_score(src_true, src_pred):.4f}")
        print(f"  Recall:    {recall_score(src_true, src_pred):.4f}")

    if tgt_true:
        print("=== Target (Spanish) side ===")
        print(f"  F1:        {f1_score(tgt_true, tgt_pred):.4f}")
        print(f"  Precision: {precision_score(tgt_true, tgt_pred):.4f}")
        print(f"  Recall:    {recall_score(tgt_true, tgt_pred):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    model_name="xlm-roberta-base",
    dataset_name=DEFAULT_DATASET,
    output_dir="./ff_model",
    epochs=10,
    batch_size=16,
    lr=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_length=512,
    seed=42,
    early_stopping_patience=3,
    hf_token=None,
):
    """
    Train a token classification model for false friend detection.

    Args:
        model_name:    HuggingFace model name (e.g. xlm-roberta-base, bert-base-multilingual-cased)
        dataset_name:  HuggingFace dataset name (default: false-friends/en_es_token)
        output_dir:    Directory to save the trained model
        epochs:        Number of training epochs
        batch_size:    Training batch size
        lr:            Learning rate
        weight_decay:  Weight decay for AdamW
        warmup_ratio:  Warmup ratio for scheduler
        max_length:    Max sequence length (source + target combined)
        seed:          Random seed
        early_stopping_patience: Early stopping patience (0 to disable)
        hf_token:      HuggingFace token for private datasets

    Returns:
        (trainer, tokenizer) tuple
    """
    print(f"Loading dataset: {dataset_name}")
    tr_sw, tr_sl, tr_tw, tr_tl = load_hf_split(dataset_name, "train", token=hf_token)
    va_sw, va_sl, va_tw, va_tl = load_hf_split(dataset_name, "test", token=hf_token)
    print(f"  Train: {len(tr_sw)} pairs, Test: {len(va_sw)} pairs")

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = FalseFriendPairDataset(tr_sw, tr_sl, tr_tw, tr_tl, tokenizer, max_length)
    val_dataset = FalseFriendPairDataset(va_sw, va_sl, va_tw, va_tl, tokenizer, max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "label_list": LABEL_LIST,
        "max_length": max_length,
    }
    with open(os.path.join(output_dir, "ff_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Final evaluation with per-side breakdown
    print("\n=== Evaluation on test set ===")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    preds_output = trainer.predict(val_dataset)
    detailed_report(preds_output.predictions, preds_output.label_ids, va_sw, va_tw)

    return trainer, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────────────
def predict(model_path, source, target):
    """
    Predict false friends in a source-target sentence pair.

    Args:
        model_path: Path to trained model directory
        source:     Source (English) sentence string
        target:     Target (Spanish) sentence string

    Returns:
        dict with keys: source_tokens, target_tokens, source_ff, target_ff
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_path = os.path.join(model_path, "ff_config.json")
    max_length = 512
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_length = config.get("max_length", 512)

    src_words = source.split()
    tgt_words = target.split()

    encoding = tokenizer(
        src_words, tgt_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    encoding_for_ids = tokenizer(
        src_words, tgt_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    input_tensors = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**input_tensors)
    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    word_ids = encoding_for_ids.word_ids()
    combined_words = src_words + tgt_words
    n_src = len(src_words)

    word_preds = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_preds:
            word_preds[word_id] = ID2LABEL[preds[token_idx]]

    src_ff, tgt_ff = [], []

    print(f"\n{'='*60}")
    print(f"Source:  {source}")
    print(f"Target:  {target}")
    print(f"{'='*60}\n")
    print(f"  {'Token':<25} {'Side':<10} {'Prediction':<10}")
    print(f"  {'-'*45}")

    for i, word in enumerate(combined_words):
        label = word_preds.get(i, "O")
        side = "source" if i < n_src else "target"
        marker = " <<<" if label == "B-FF" else ""
        print(f"  {word:<25} {side:<10} {label:<10}{marker}")
        if label == "B-FF":
            (src_ff if i < n_src else tgt_ff).append(word)

    print()
    if src_ff or tgt_ff:
        if src_ff:
            print(f"  Source false friends: {src_ff}")
        if tgt_ff:
            print(f"  Target false friends: {tgt_ff}")
    else:
        print("  No false friends detected.")

    return {
        "source_tokens": src_words,
        "target_tokens": tgt_words,
        "source_ff": src_ff,
        "target_ff": tgt_ff,
        "predictions": {i: word_preds.get(i, "O") for i in range(len(combined_words))},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path, dataset_name=DEFAULT_DATASET, split="test", hf_token=None):
    """
    Evaluate a trained model on a dataset split.

    Args:
        model_path:    Path to trained model directory
        dataset_name:  HuggingFace dataset name (default: false-friends/en_es_token)
        split:         Dataset split to evaluate on (default: test)
        hf_token:      HuggingFace token for private datasets
    """
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    config_path = os.path.join(model_path, "ff_config.json")
    max_length = 512
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_length = config.get("max_length", 512)

    print(f"Loading dataset: {dataset_name} [{split}]")
    src_w, src_l, tgt_w, tgt_l = load_hf_split(dataset_name, split, token=hf_token)
    print(f"  Evaluating on {len(src_w)} sentence pairs")

    dataset = FalseFriendPairDataset(src_w, src_l, tgt_w, tgt_l, tokenizer, max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    preds_output = trainer.predict(dataset)
    detailed_report(preds_output.predictions, preds_output.label_ids, src_w, tgt_w)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="False Friend Token Classification (Paired)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train ---
    tp = subparsers.add_parser("train", help="Train a token classification model")
    tp.add_argument("--model_name", type=str, default="xlm-roberta-base")
    tp.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    tp.add_argument("--output_dir", type=str, default="./ff_model")
    tp.add_argument("--epochs", type=int, default=10)
    tp.add_argument("--batch_size", type=int, default=16)
    tp.add_argument("--lr", type=float, default=5e-5)
    tp.add_argument("--weight_decay", type=float, default=0.01)
    tp.add_argument("--warmup_ratio", type=float, default=0.1)
    tp.add_argument("--max_length", type=int, default=512)
    tp.add_argument("--seed", type=int, default=42)
    tp.add_argument("--early_stopping_patience", type=int, default=3)
    tp.add_argument("--hf_token", type=str, default=None)

    # --- Predict ---
    pp = subparsers.add_parser("predict", help="Predict false friends in a sentence pair")
    pp.add_argument("--model_path", type=str, required=True)
    pp.add_argument("--source", type=str, required=True)
    pp.add_argument("--target", type=str, required=True)

    # --- Evaluate ---
    ep = subparsers.add_parser("evaluate", help="Evaluate model on a dataset")
    ep.add_argument("--model_path", type=str, required=True)
    ep.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    ep.add_argument("--split", type=str, default="test")
    ep.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_length=args.max_length,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            hf_token=args.hf_token,
        )
    elif args.command == "predict":
        predict(args.model_path, args.source, args.target)
    elif args.command == "evaluate":
        evaluate_model(args.model_path, args.dataset_name, args.split, args.hf_token)


if __name__ == "__main__":
    main()
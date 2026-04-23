"""
Token Classification for False Friend Detection using HuggingFace Transformers.

Source and target sentences are concatenated into a single sequence:
  [CLS] source_tokens [SEP] target_tokens [SEP]
so the model can attend across both languages and learn cross-lingual
mismatch patterns that define false friends.

Labels are aligned for both source and target tokens; special tokens get -100.

Usage:
  Train:
    python token_classification.py train \
        --model_name xlm-roberta-base \
        --data false_friend_ner.csv \
        --output_dir ./ff_model \
        --epochs 10 --batch_size 16 --lr 5e-5

  Predict:
    python token_classification.py predict \
        --model_path ./ff_model \
        --source "This is a sensible solution" \
        --target "Esta es una solución sensible"

  Evaluate:
    python token_classification.py evaluate \
        --model_path ./ff_model \
        --data false_friend_ner.csv
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
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
# Label scheme
# ──────────────────────────────────────────────────────────────────────────────
LABEL_LIST = ["O", "B-FF"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — concatenated source + target
# ──────────────────────────────────────────────────────────────────────────────
class FalseFriendPairDataset(Dataset):
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

        # Concatenate as sentence-pair: tokenizer handles [CLS]...[SEP]...[SEP]
        encoding = self.tokenizer(
            s_words,
            t_words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # Build combined label list: source labels then target labels
        combined_labels = s_labels + t_labels

        # Align subword tokens to word-level labels
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], padding)
                label_ids.append(-100)
            elif word_id != previous_word_id:
                if word_id < len(combined_labels):
                    label_ids.append(LABEL2ID[combined_labels[word_id]])
                else:
                    label_ids.append(-100)
            else:
                # Sub-token continuation
                label_ids.append(-100)
            previous_word_id = word_id

        encoding["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_csv_pairs(csv_path):
    """Load the NER CSV and group into (source, target) pairs by sentence_id.
    Returns four lists: src_words, src_labels, tgt_words, tgt_labels."""
    df = pd.read_csv(csv_path)

    src_words, src_labels, tgt_words, tgt_labels = [], [], [], []

    for sid, group in df.groupby("sentence_id", sort=True):
        src_group = group[group["side"] == "source"]
        tgt_group = group[group["side"] == "target"]
        if src_group.empty or tgt_group.empty:
            continue
        src_words.append(src_group["words"].astype(str).tolist())
        src_labels.append(src_group["labels"].tolist())
        tgt_words.append(tgt_group["words"].astype(str).tolist())
        tgt_labels.append(tgt_group["labels"].tolist())

    return src_words, src_labels, tgt_words, tgt_labels


def split_data(src_w, src_l, tgt_w, tgt_l, test_size=0.2, seed=42):
    """Stratified split based on whether *either* side has B-FF."""
    has_ff = [
        1 if ("B-FF" in sl or "B-FF" in tl) else 0
        for sl, tl in zip(src_l, tgt_l)
    ]
    indices = list(range(len(src_w)))
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=has_ff
    )
    pick = lambda lst, idxs: [lst[i] for i in idxs]
    return (
        pick(src_w, train_idx), pick(src_l, train_idx),
        pick(tgt_w, train_idx), pick(tgt_l, train_idx),
        pick(src_w, val_idx), pick(src_l, val_idx),
        pick(tgt_w, val_idx), pick(tgt_l, val_idx),
    )


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

        word_idx = 0
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(ID2LABEL[l])
            seq_preds.append(ID2LABEL[p])
            word_idx += 1

        all_true.append(seq_labels)
        all_pred.append(seq_preds)

        # Split into source / target portions
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
def train(args):
    print(f"Loading data from {args.data}")
    src_w, src_l, tgt_w, tgt_l = load_csv_pairs(args.data)
    print(f"Loaded {len(src_w)} sentence pairs")

    (tr_sw, tr_sl, tr_tw, tr_tl,
     va_sw, va_sl, va_tw, va_tl) = split_data(
        src_w, src_l, tgt_w, tgt_l,
        test_size=args.test_size, seed=args.seed
    )
    print(f"Train: {len(tr_sw)}, Val: {len(va_sw)}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = FalseFriendPairDataset(tr_sw, tr_sl, tr_tw, tr_tl, tokenizer, args.max_length)
    val_dataset = FalseFriendPairDataset(va_sw, va_sl, va_tw, va_tl, tokenizer, args.max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

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

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "model_name": args.model_name,
        "label_list": LABEL_LIST,
        "max_length": args.max_length,
    }
    with open(os.path.join(args.output_dir, "ff_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Final evaluation with per-side breakdown
    print("\n=== Evaluation on validation set ===")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    preds_output = trainer.predict(val_dataset)
    detailed_report(preds_output.predictions, preds_output.label_ids, va_sw, va_tw)


# ──────────────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────────────
def predict(args):
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_path = os.path.join(args.model_path, "ff_config.json")
    max_length = 512
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_length = config.get("max_length", 512)

    src_words = args.source.split()
    tgt_words = args.target.split()

    # Tokenise as sentence pair
    encoding = tokenizer(
        src_words,
        tgt_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    encoding_for_ids = tokenizer(
        src_words,
        tgt_words,
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

    # Map word_id → prediction (first subtoken wins)
    word_preds = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_preds:
            word_preds[word_id] = ID2LABEL[preds[token_idx]]

    print(f"\n{'='*60}")
    print(f"Source:  {args.source}")
    print(f"Target:  {args.target}")
    print(f"{'='*60}\n")

    src_ff, tgt_ff = [], []

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


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(args):
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_path = os.path.join(args.model_path, "ff_config.json")
    max_length = 512
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_length = config.get("max_length", 512)

    src_w, src_l, tgt_w, tgt_l = load_csv_pairs(args.data)
    print(f"Evaluating on {len(src_w)} sentence pairs")

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
    tp.add_argument("--model_name", type=str, default="xlm-roberta-base",
                    help="HuggingFace model name (e.g. xlm-roberta-base, bert-base-multilingual-cased)")
    tp.add_argument("--data", type=str, required=True, help="Path to NER CSV file")
    tp.add_argument("--output_dir", type=str, default="./ff_model")
    tp.add_argument("--epochs", type=int, default=10)
    tp.add_argument("--batch_size", type=int, default=16)
    tp.add_argument("--lr", type=float, default=5e-5)
    tp.add_argument("--weight_decay", type=float, default=0.01)
    tp.add_argument("--warmup_ratio", type=float, default=0.1)
    tp.add_argument("--max_length", type=int, default=512)
    tp.add_argument("--test_size", type=float, default=0.2)
    tp.add_argument("--seed", type=int, default=42)
    tp.add_argument("--early_stopping_patience", type=int, default=3,
                    help="Early stopping patience (0 to disable)")

    # --- Predict ---
    pp = subparsers.add_parser("predict", help="Predict false friends in a sentence pair")
    pp.add_argument("--model_path", type=str, required=True)
    pp.add_argument("--source", type=str, required=True, help="Source (English) sentence")
    pp.add_argument("--target", type=str, required=True, help="Target (Spanish) sentence")

    # --- Evaluate ---
    ep = subparsers.add_parser("evaluate", help="Evaluate model on a dataset")
    ep.add_argument("--model_path", type=str, required=True)
    ep.add_argument("--data", type=str, required=True, help="Path to NER CSV file")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()

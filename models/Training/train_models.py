#!/usr/bin/env python3
"""
~/MGW/models/Training/train_models.py

Unified training script for both sub-models.

Usage
-----
    python train_models.py [--header] [--body] [--all]

If no flag is given, --all is assumed.
"""

import os
import sys
import json
import argparse
import logging
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
MGW_ROOT     = Path.home() / "MGW"
DATASET_DIR  = Path.home() / "Datasets"
TRAINING_DIR = MGW_ROOT / "models" / "Training"
HEADER_DIR   = MGW_ROOT / "models" / "Header"
BODY_DIR     = MGW_ROOT / "models" / "Body"

TRAINING_DIR.mkdir(parents=True, exist_ok=True)
HEADER_DIR.mkdir(parents=True, exist_ok=True)
BODY_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = TRAINING_DIR / "training.log"

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("trainer")

# ─── Dependency bootstrap ─────────────────────────────────────────────────────
REQUIRED = [
    "xgboost", "numpy", "pandas", "scikit-learn",
    "transformers", "torch", "accelerate",
    "matplotlib", "seaborn",
]

def _ensure_deps():
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            logger.info(f"Installing: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                                   "--quiet", "--break-system-packages"])

_ensure_deps()

import numpy  as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ══════════════════════════════════════════════════════════════════════════════
# Feature column definitions (mirrors header.py / body.py)
# ══════════════════════════════════════════════════════════════════════════════
HEADER_COLS = [
    "has_from", "has_to", "has_cc", "has_bcc",
    "has_subject", "subject_length", "subject_word_count",
    "subject_has_reply", "subject_has_fwd", "subject_has_urgent",
    "subject_has_alert", "subject_has_verify", "subject_all_caps",
    "subject_caps_ratio", "subject_has_numbers", "subject_has_special",
    "subject_exclamation", "subject_question", "subject_money",
    "has_date", "has_message_id", "has_reply_to", "has_return_path", "has_dkim",
]

BODY_COLS = [
    "body_length", "body_word_count", "body_line_count", "body_paragraph_count",
    "uppercase_count", "lowercase_count", "digit_count", "space_count",
    "punctuation_count", "special_char_count",
    "uppercase_ratio", "lowercase_ratio", "digit_ratio", "space_ratio",
    "punctuation_ratio", "special_char_ratio",
    "avg_word_length", "unique_word_count", "unique_word_ratio",
    "url_count", "unique_url_count", "url_avg_length", "url_max_length",
    "url_has_ip", "url_has_port", "url_has_https", "url_has_http",
    "url_count_https", "url_count_http", "url_suspicious_tlds",
    "url_has_subdomains", "url_max_dots", "url_avg_slashes",
    "url_has_at_symbol", "url_has_double_slash", "url_has_hyphen",
    "url_has_underscore", "url_has_percent_encoding",
    "email_in_body_count", "unique_email_in_body_count",
    "phone_count", "ip_address_count",
    "dollar_sign_count", "euro_sign_count", "pound_sign_count", "yen_sign_count",
    "total_money_symbols", "exclamation_count", "question_count",
    "exclamation_ratio", "question_ratio",
    "has_html_tags", "html_tag_count",
    "keyword_urgent", "keyword_verify", "keyword_account", "keyword_bank",
    "keyword_paypal", "keyword_suspended", "keyword_click", "keyword_login",
    "keyword_password", "keyword_credit", "keyword_social_security",
    "keyword_ssn", "keyword_limited", "keyword_unusual", "keyword_activity",
    "keyword_confirm", "keyword_update", "keyword_security", "keyword_fraud",
    "keyword_claim", "keyword_prize", "keyword_winner", "keyword_lottery",
    "keyword_inheritance", "keyword_million", "keyword_billion",
    "keyword_dollars", "keyword_transfer", "keyword_western_union",
    "keyword_money_gram", "keyword_wire_transfer", "keyword_bank_account",
    "keyword_routing_number", "keyword_credit_card", "keyword_debit_card",
    "keyword_expire", "keyword_deadline",
    "total_phishing_keywords", "unique_phishing_keywords",
    "urgency_score", "fear_score", "curiosity_score",
    "has_html_entities", "html_entity_count",
    "has_javascript", "has_onclick", "has_onload",
    "has_form", "has_input",
    "attachment_count", "has_executable_attachment", "has_archive_attachment",
    "has_pdf_attachment", "has_document_attachment", "has_image_attachment",
]

# ══════════════════════════════════════════════════════════════════════════════
# Dataset loader
# ══════════════════════════════════════════════════════════════════════════════
def load_datasets() -> pd.DataFrame:
    csv_files = list(DATASET_DIR.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {DATASET_DIR}!")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            logger.info(f"Loaded {f.name}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipped {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total combined dataset: {len(combined)} rows")
    return combined


def _save_metrics(metrics: dict, name: str):
    out = TRAINING_DIR / f"{name}_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info(f"Metrics saved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ── Train Header XGBoost ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def train_header(df: pd.DataFrame):
    logger.info("=== Training Header XGBoost Model ===")

    available = [c for c in HEADER_COLS if c in df.columns]
    if not available:
        logger.error("No header feature columns found in dataset!")
        return

    X = df[available].fillna(0).astype(float)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="auc",
        verbosity=0,
        random_state=42,
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train,
              eval_set=eval_set,
              verbose=False)

    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)[:, 1]
    roc_auc      = roc_auc_score(y_test, y_proba)
    avg_prec     = average_precision_score(y_test, y_proba)
    report       = classification_report(y_test, y_pred, output_dict=True)
    cm           = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"Header ROC-AUC: {roc_auc:.4f}  |  Avg Precision: {avg_prec:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Save model
    model_path = HEADER_DIR / "xgb_header_model.json"
    model.save_model(str(model_path))
    logger.info(f"Header model saved to {model_path}")

    # Feature importance plot
    _plot_feature_importance(model.feature_importances_, available, "header")

    _save_metrics({
        "model": "xgboost_header",
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "classification_report": report,
        "confusion_matrix": cm,
        "features_used": available,
    }, "header")


def _plot_feature_importance(importances, feature_names, tag: str):
    try:
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
        names, vals = zip(*pairs)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(names[::-1], vals[::-1], color="#1a6faf")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-20 Feature Importances ({tag})")
        plt.tight_layout()
        out = TRAINING_DIR / f"{tag}_feature_importance.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        logger.info(f"Feature importance plot saved to {out}")
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── Train Body RoBERTa ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
ROBERTA_MODEL_NAME = "roberta-base"
MAX_LEN            = 512
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


class _EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def train_body(df: pd.DataFrame):
    logger.info("=== Training Body RoBERTa Model ===")

    # Determine text column
    text_col = next(
        (c for c in ["body_text", "body", "text", "email_text", "message"] if c in df.columns),
        None
    )
    if text_col is None:
        kw_cols = [c for c in df.columns if c.startswith("keyword_")]
        if kw_cols:
            logger.info("No raw text column found; synthesising from keyword columns.")
            df = df.copy()
            df["_synth_text"] = df[kw_cols].apply(
                lambda row: " ".join(k.replace("keyword_", "") for k, v in row.items() if v),
                axis=1
            )
            text_col = "_synth_text"
        else:
            logger.error("No text column available for body training. Skipping.")
            return

    data = df[[text_col, "label"]].dropna()
    X    = data[text_col].tolist()
    y    = data["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    logger.info(f"Training on {len(X_train)} samples (device={DEVICE})")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME, num_labels=2
    )

    def _encode(texts):
        return tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )

    train_enc = _encode(X_train)
    test_enc  = _encode(X_test)

    train_ds = _EmailDataset(train_enc, y_train)
    test_ds  = _EmailDataset(test_enc,  y_test)

    save_path = str(BODY_DIR / "roberta_body_model")

    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(TRAINING_DIR / "roberta_logs"),
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Body RoBERTa model saved to {save_path}")

    # Evaluate
    preds_out = trainer.predict(test_ds)
    logits    = preds_out.predictions
    probs     = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_pred    = (probs >= 0.5).astype(int)
    roc_auc   = roc_auc_score(y_test, probs)
    avg_prec  = average_precision_score(y_test, probs)
    report    = classification_report(y_test, y_pred, output_dict=True)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"Body ROC-AUC: {roc_auc:.4f}  |  Avg Precision: {avg_prec:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    _save_metrics({
        "model": "roberta_body",
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "classification_report": report,
        "confusion_matrix": cm,
    }, "body")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train MGW email filter models")
    parser.add_argument("--header", action="store_true", help="Train header XGBoost model")
    parser.add_argument("--body",   action="store_true", help="Train body RoBERTa model")
    parser.add_argument("--all",    action="store_true", help="Train both models (default)")
    args = parser.parse_args()

    if not (args.header or args.body or args.all):
        args.all = True

    df = load_datasets()
    if "label" not in df.columns:
        logger.error("Dataset must contain a 'label' column (0=legit, 1=phishing).")
        sys.exit(1)

    label_counts = df["label"].value_counts().to_dict()
    logger.info(f"Label distribution: {label_counts}")

    if args.header or args.all:
        train_header(df)

    if args.body or args.all:
        train_body(df)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

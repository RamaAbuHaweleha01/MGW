#!/usr/bin/env python3
"""
~/MGW/models/Training/train_models.py
Unified training script — Header XGBoost, Body RoBERTa, Subject TF-IDF.

Usage
-----
    python train_models.py [--header] [--body] [--subject] [--all]

If no flag given, --all is assumed.

Models produced
---------------
  ~/MGW/models/Header/xgb_header_model.json          XGBoost (structural + auth)
  ~/MGW/models/Header/tfidf_subject_header.pkl        TF-IDF  (subject NLP)
  ~/MGW/models/Body/roberta_finetuned/                RoBERTa (body NLP)
  ~/MGW/models/Body/tfidf_fallback.pkl                TF-IDF  (body NLP fallback)
  ~/MGW/models/Body/tfidf_subject.pkl                 TF-IDF  (subject — body module copy)
"""
from __future__ import annotations
import os, sys, json, argparse, logging, importlib, subprocess, pickle
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
MGW_ROOT     = Path.home() / "MGW"
DATASET_DIR  = Path.home() / "Datasets"
TRAINING_DIR = MGW_ROOT / "models" / "Training"
HEADER_DIR   = MGW_ROOT / "models" / "Header"
BODY_DIR     = MGW_ROOT / "models" / "Body"
for d in [TRAINING_DIR, HEADER_DIR, BODY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = TRAINING_DIR / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")

# ─── Dependency bootstrap ─────────────────────────────────────────────────────
REQUIRED = [
    "xgboost","numpy","pandas","scikit-learn",
    "transformers","torch","accelerate",
    "matplotlib","seaborn","psutil",
]

def _ensure_deps():
    for pkg in REQUIRED:
        mod = pkg.replace("-","_")
        try:
            importlib.import_module(mod)
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
import psutil    # resource monitoring
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)

# ══════════════════════════════════════════════════════════════════════════════
# Feature column definitions (keep in sync with header.py / body.py)
# ══════════════════════════════════════════════════════════════════════════════
HEADER_COLS = [
    "has_from","has_to","has_cc","has_bcc",
    "has_subject","subject_length","subject_word_count",
    "subject_has_reply","subject_has_fwd",
    "subject_has_urgent","subject_has_alert","subject_has_verify",
    "subject_all_caps","subject_caps_ratio",
    "subject_has_numbers","subject_has_special",
    "subject_exclamation","subject_money",
    "has_date","has_message_id","has_reply_to","has_return_path","has_dkim",
    "spf_fail","dkim_fail","dmarc_fail",
    "domain_mismatch","suspicious_tld_sender","has_numeric_in_domain",
    "received_hops","date_is_future","date_is_weekend","hour_sent",
    "dollar_count","total_money_symbols",
    "url_count","url_has_ip","url_suspicious_tlds","url_mismatch_count",
    "has_script","has_iframe","has_form",
]

BODY_COLS = [
    "body_length","body_word_count","body_line_count","body_paragraph_count",
    "avg_word_length","unique_word_count","unique_word_ratio","caps_ratio",
    "url_count","url_avg_length","url_max_length","url_has_ip",
    "url_count_https","url_count_http","url_suspicious_tlds",
    "url_has_subdomains","url_max_dots","url_has_percent_encoding",
    "email_in_body_count","unique_email_in_body_count",
    "phone_count","ip_address_count",
    "dollar_sign_count","euro_sign_count","pound_sign_count","yen_sign_count",
    "total_money_symbols","exclamation_count","question_count",
    "has_html_tags","html_tag_count","html_entity_count","has_html_entities",
    "has_script","has_onclick","has_onload","has_form","has_input_password",
    "has_eval","has_base64","has_data_uri","has_javascript","obfuscated_chars",
    "keyword_urgent","keyword_verify","keyword_account","keyword_bank",
    "keyword_paypal","keyword_suspended","keyword_click","keyword_login",
    "keyword_password","keyword_credit","keyword_social_security",
    "keyword_ssn","keyword_limited","keyword_unusual","keyword_activity",
    "keyword_confirm","keyword_update","keyword_security","keyword_fraud",
    "keyword_claim","keyword_prize","keyword_winner","keyword_lottery",
    "keyword_inheritance","keyword_million","keyword_billion",
    "keyword_dollars","keyword_transfer","keyword_western_union",
    "keyword_money_gram","keyword_wire_transfer","keyword_bank_account",
    "keyword_routing_number","keyword_credit_card","keyword_debit_card",
    "keyword_expire","keyword_deadline",
    "total_phishing_keywords","unique_phishing_keywords",
    "urgency_score","fear_score","curiosity_score",
    # Attachment features (present in dataset when pre-extracted)
    "attachment_count","has_executable_attachment","has_archive_attachment",
    "has_pdf_attachment","has_document_attachment","has_image_attachment",
    "has_office_macro_attachment","has_script_attachment",
]


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loader
# ══════════════════════════════════════════════════════════════════════════════
def load_datasets() -> pd.DataFrame:
    csvs = list(DATASET_DIR.glob("*.csv"))
    if not csvs:
        logger.error(f"No CSV files found in {DATASET_DIR}"); sys.exit(1)

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False)
            logger.info(f"Loaded {f.name}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipped {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows")
    return combined


def _normalise_label(df: pd.DataFrame) -> pd.DataFrame:
    lcol = next((c for c in ["label","Label","spam","class"] if c in df.columns), None)
    if lcol is None:
        logger.error("No label column found."); sys.exit(1)
    if lcol != "label":
        df = df.rename(columns={lcol: "label"})
    df["label"] = df["label"].astype(int)
    return df


def _save_metrics(metrics: dict, name: str):
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = TRAINING_DIR / f"{name}_metrics_{ts}.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info(f"Metrics → {out}")


def _plot_importance(importances, names, tag: str):
    try:
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:20]
        ns, vs = zip(*pairs)
        fig, ax = plt.subplots(figsize=(10,7))
        ax.barh(ns[::-1], vs[::-1], color="#1a6faf")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-20 Features ({tag})")
        plt.tight_layout()
        out = TRAINING_DIR / f"{tag}_feature_importance.png"
        plt.savefig(str(out), dpi=150); plt.close()
        logger.info(f"Plot → {out}")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Subject TF-IDF (used by both header.py and body.py)
# ══════════════════════════════════════════════════════════════════════════════
def train_subject_tfidf(df: pd.DataFrame):
    logger.info("=== Training Subject TF-IDF ===")
    subj_col = next((c for c in ["subject","Subject"] if c in df.columns), None)
    if subj_col is None:
        logger.warning("No subject column — skipping subject TF-IDF")
        return

    data = df[[subj_col,"label"]].dropna()
    X = data[subj_col].astype(str).tolist()
    y = data["label"].astype(int).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    pl = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                   stop_words="english", sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=1.0,
                                      class_weight="balanced",
                                      solver="lbfgs", random_state=42)),
    ])
    pl.fit(X_tr, y_tr)

    y_prob = pl.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    logger.info(f"Subject TF-IDF ROC-AUC={auc:.4f}")

    # Save to both Header and Body dirs (each model loads its own copy)
    for dest in [HEADER_DIR / "tfidf_subject_header.pkl",
                 BODY_DIR   / "tfidf_subject.pkl"]:
        with open(dest, "wb") as f:
            pickle.dump(pl, f)
        logger.info(f"Subject TF-IDF saved → {dest}")

    _save_metrics({
        "model":   "tfidf_subject",
        "roc_auc": auc,
        "train_n": len(X_tr),
        "test_n":  len(X_te),
    }, "subject_tfidf")


# ══════════════════════════════════════════════════════════════════════════════
# Header XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def train_header(df: pd.DataFrame):
    logger.info("=== Training Header XGBoost ===")
    available = [c for c in HEADER_COLS if c in df.columns]
    if not available:
        logger.error("No header feature columns found."); return

    X = df[available].fillna(0).astype(float)
    y = df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg / max(pos, 1),
        use_label_encoder=False, eval_metric="auc",
        verbosity=0, random_state=42, early_stopping_rounds=20,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_prob   = clf.predict_proba(X_te)[:, 1]
    y_pred   = (y_prob >= 0.5).astype(int)
    roc_auc  = roc_auc_score(y_te, y_prob)
    avg_prec = average_precision_score(y_te, y_prob)
    logger.info(f"Header XGBoost ROC-AUC={roc_auc:.4f}  AvgPrec={avg_prec:.4f}")
    logger.info("\n" + classification_report(y_te, y_pred))

    model_path = HEADER_DIR / "xgb_header_model.json"
    clf.save_model(str(model_path))
    logger.info(f"Header model saved → {model_path}")
    _plot_importance(clf.feature_importances_, available, "header")
    _save_metrics({
        "model": "xgboost_header",
        "roc_auc": roc_auc, "avg_precision": avg_prec,
        "classification_report": classification_report(y_te, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "features_used": available,
    }, "header")


# ══════════════════════════════════════════════════════════════════════════════
# Body RoBERTa + TF-IDF fallback
# ══════════════════════════════════════════════════════════════════════════════
ROBERTA_MODEL_NAME = "roberta-base"
MAX_LEN            = 512
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


class _EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings; self.labels = labels
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self): return len(self.labels)


def _check_resources() -> dict:
    """Log CPU/RAM before heavy training."""
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    logger.info(
        f"System resources — CPU={cpu:.1f}%  "
        f"RAM={ram.percent:.1f}%  "
        f"Available={ram.available/1e9:.1f}GB"
    )
    return {"cpu": cpu, "ram_pct": ram.percent, "ram_avail_gb": ram.available/1e9}


def train_body(df: pd.DataFrame):
    logger.info("=== Training Body RoBERTa ===")
    _check_resources()

    text_col = next(
        (c for c in ["body_text","body","text","email_text","message","clean_text"]
         if c in df.columns), None)
    if text_col is None:
        kw_cols = [c for c in df.columns if c.startswith("keyword_")]
        if kw_cols:
            logger.info("No raw text column — synthesising from keyword columns")
            df = df.copy()
            df["_synth"] = df[kw_cols].apply(
                lambda r: " ".join(k.replace("keyword_","") for k,v in r.items() if v),
                axis=1)
            text_col = "_synth"
        else:
            logger.error("No text column available."); return

    data   = df[[text_col,"label"]].dropna()
    X      = data[text_col].astype(str).tolist()
    y      = data["label"].astype(int).tolist()

    # Also train TF-IDF fallback first (lightweight)
    logger.info("Training TF-IDF body fallback …")
    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)
    body_tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1,3),
                                   stop_words="english", sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                      class_weight="balanced",
                                      solver="lbfgs", random_state=42)),
    ])
    body_tfidf.fit(X_tr_t, y_tr_t)
    t_auc = roc_auc_score(y_te_t, body_tfidf.predict_proba(X_te_t)[:,1])
    logger.info(f"Body TF-IDF ROC-AUC={t_auc:.4f}")
    tfidf_path = BODY_DIR / "tfidf_fallback.pkl"
    with open(tfidf_path, "wb") as f:
        pickle.dump(body_tfidf, f)
    logger.info(f"Body TF-IDF saved → {tfidf_path}")

    # RoBERTa
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    logger.info(f"RoBERTa training on {len(X_tr)} samples (device={DEVICE})")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME, num_labels=2)

    def _enc(texts):
        return tokenizer(texts, truncation=True, padding=True,
                         max_length=MAX_LEN, return_tensors="pt")

    train_ds = _EmailDataset(_enc(X_tr), y_tr)
    test_ds  = _EmailDataset(_enc(X_te),  y_te)
    save_path = str(BODY_DIR / "roberta_finetuned")

    args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(TRAINING_DIR / "roberta_logs"),
        logging_steps=100, report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"RoBERTa saved → {save_path}")

    preds_out = trainer.predict(test_ds)
    probs     = torch.softmax(torch.tensor(preds_out.predictions), dim=-1).numpy()[:, 1]
    y_pred    = (probs >= 0.5).astype(int)
    roc_auc   = roc_auc_score(y_te, probs)
    avg_prec  = average_precision_score(y_te, probs)
    logger.info(f"Body RoBERTa ROC-AUC={roc_auc:.4f}  AvgPrec={avg_prec:.4f}")
    logger.info("\n" + classification_report(y_te, y_pred))

    _save_metrics({
        "model": "roberta_body", "roc_auc": roc_auc, "avg_precision": avg_prec,
        "classification_report": classification_report(y_te, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "tfidf_roc_auc": t_auc,
    }, "body")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Train MGW email filter models")
    ap.add_argument("--header",  action="store_true", help="Train header XGBoost")
    ap.add_argument("--body",    action="store_true", help="Train body RoBERTa + TF-IDF")
    ap.add_argument("--subject", action="store_true", help="Train subject TF-IDF")
    ap.add_argument("--all",     action="store_true", help="Train all models (default)")
    args = ap.parse_args()
    if not (args.header or args.body or args.subject or args.all):
        args.all = True

    df = load_datasets()
    df = _normalise_label(df)
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    if args.subject or args.all:
        train_subject_tfidf(df)

    if args.header or args.all:
        train_header(df)

    if args.body or args.all:
        train_body(df)

    logger.info("All training complete.")


if __name__ == "__main__":
    main()

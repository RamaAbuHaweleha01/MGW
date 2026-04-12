#!/usr/bin/env python3
"""~/MGW/models/Body/body.py — RoBERTa body risk analyzer."""
from __future__ import annotations
import os, sys, json, logging, subprocess, pickle, math
from datetime import datetime
from pathlib import Path

MGW_ROOT    = Path.home() / "MGW"
MODEL_DIR   = MGW_ROOT / "models" / "Body"
ROBERTA_DIR = MODEL_DIR / "roberta_finetuned"
TFIDF_FILE  = MODEL_DIR / "tfidf_fallback.pkl"
LOG_FILE    = MODEL_DIR / "body.log"
DATASET_DIR = Path.home() / "Datasets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("body_analyzer")
if not logger.handlers:
    h = logging.FileHandler(LOG_FILE)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(h)

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

REQUIRED = [("numpy","numpy"),("pandas","pandas"),("scikit-learn","sklearn"),
            ("transformers","transformers"),("torch","torch"),("accelerate","accelerate")]
for _pkg, _mod in REQUIRED:
    try: __import__(_mod)
    except ImportError:
        logger.info(f"Installing {_pkg}")
        _pip(_pkg)

import numpy  as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ─── RoBERTa ─────────────────────────────────────────────────────────────────
ROBERTA_BASE = "roberta-base"
MAX_LEN      = 512
DEVICE       = "cpu"
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    pass

_ROBERTA_MODEL     = None
_ROBERTA_TOKENIZER = None
_TFIDF_PIPELINE    = None

def _load_roberta():
    global _ROBERTA_MODEL, _ROBERTA_TOKENIZER
    if _ROBERTA_MODEL is not None:
        return _ROBERTA_TOKENIZER, _ROBERTA_MODEL

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    if ROBERTA_DIR.exists():
        logger.info("Loading fine-tuned RoBERTa")
        tok   = AutoTokenizer.from_pretrained(str(ROBERTA_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(ROBERTA_DIR))
    else:
        logger.info("Loading base RoBERTa (not fine-tuned yet)")
        tok   = AutoTokenizer.from_pretrained(ROBERTA_BASE)
        model = AutoModelForSequenceClassification.from_pretrained(
            ROBERTA_BASE, num_labels=2)
        _finetune_roberta(tok, model)

    model.to(DEVICE).eval()
    _ROBERTA_TOKENIZER = tok
    _ROBERTA_MODEL     = model
    return tok, model


def _finetune_roberta(tok, model):
    """Fine-tune RoBERTa on master dataset if available."""
    import torch
    from torch.utils.data import Dataset
    from transformers import TrainingArguments, Trainer

    master = DATASET_DIR / "master_phishing_dataset.csv"
    if not master.exists():
        logger.warning("No master dataset — RoBERTa stays as base model")
        return

    df = pd.read_csv(master, low_memory=False)
    if "label" not in df.columns:
        return

    text_col = next((c for c in ["body","text","subject","message"]
                     if c in df.columns), None)
    if text_col is None:
        return

    df = df[[text_col,"label"]].dropna()
    # Limit to 50k for speed — use stratified sample
    if len(df) > 50000:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), 25000), random_state=42))

    X = df[text_col].astype(str).tolist()
    y = df["label"].astype(int).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    class EmailDS(Dataset):
        def __init__(self, texts, labels):
            enc = tok(texts, truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt")
            self.ids    = enc["input_ids"]
            self.masks  = enc["attention_mask"]
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):  return len(self.labels)
        def __getitem__(self, i):
            return {"input_ids":      self.ids[i],
                    "attention_mask": self.masks[i],
                    "labels":         self.labels[i]}

    train_ds = EmailDS(X_tr, y_tr)
    eval_ds  = EmailDS(X_te, y_te)

    args = TrainingArguments(
        output_dir=str(ROBERTA_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50, report_to="none",
        fp16=(DEVICE=="cuda"),
    )
    Trainer(model=model, args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds).train()

    model.save_pretrained(str(ROBERTA_DIR))
    tok.save_pretrained(str(ROBERTA_DIR))
    logger.info(f"RoBERTa fine-tuned and saved → {ROBERTA_DIR}")


def _roberta_score(clean_text: str) -> float:
    import torch
    tok, model = _load_roberta()
    enc = tok(clean_text, truncation=True,
               max_length=MAX_LEN, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    prob = torch.softmax(logits, dim=-1).cpu().numpy()[0][1]
    return float(prob)


def _load_tfidf():
    global _TFIDF_PIPELINE
    if _TFIDF_PIPELINE is not None:
        return _TFIDF_PIPELINE

    if TFIDF_FILE.exists():
        with open(TFIDF_FILE,"rb") as f:
            _TFIDF_PIPELINE = pickle.load(f)
        return _TFIDF_PIPELINE

    master = DATASET_DIR / "master_phishing_dataset.csv"
    if not master.exists():
        return None

    df = pd.read_csv(master, low_memory=False)
    text_col = next((c for c in ["body","text","subject"] if c in df.columns), None)
    if not text_col or "label" not in df.columns:
        return None

    df = df[[text_col,"label"]].dropna()
    X  = df[text_col].astype(str).tolist()
    y  = df["label"].astype(int).tolist()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.1,
                                         random_state=42, stratify=y)
    pl = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1,3),
                                   stop_words="english", sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                      class_weight="balanced",
                                      solver="lbfgs", random_state=42)),
    ])
    pl.fit(X_tr, y_tr)
    with open(TFIDF_FILE,"wb") as f:
        pickle.dump(pl, f)
    logger.info("TF-IDF fallback model trained")
    _TFIDF_PIPELINE = pl
    return pl


# ─── Semantic metadata scoring ────────────────────────────────────────────────
SEMANTIC_WEIGHTS = {
    "url_has_ip":             0.35,
    "url_mismatch_count":     0.40,
    "url_suspicious_tlds":    0.30,
    "has_script":             0.30,
    "has_iframe":             0.30,
    "has_eval":               0.35,
    "has_base64":             0.25,
    "has_data_uri":           0.30,
    "has_form":               0.20,
    "has_input_password":     0.35,
    "domain_mismatch":        0.35,
    "spf_fail":               0.35,
    "dmarc_fail":             0.40,
    "urgency_score":          0.30,
    "fear_score":             0.30,
    "curiosity_score":        0.20,
    "total_phishing_keywords":0.04,  # per-count
    "html_entity_count":      0.02,  # per-count (obfuscation)
    "date_is_future":         0.25,
    "has_dkim":              -0.20,  # legitimacy
}

def _semantic_score(meta: dict):
    positive = negative = 0.0
    factors  = []
    for feat, weight in SEMANTIC_WEIGHTS.items():
        val = float(meta.get(feat, 0) or 0)
        if not val:
            continue
        c = weight * val
        if c > 0: positive += c
        else:     negative += abs(c)
        factors.append(f"{feat}={val:.2f} w={weight:+.2f}")
    net  = positive - negative
    prob = (math.tanh(net * 0.7) + 1) / 2
    return float(np.clip(prob, 0.0, 1.0)), factors


# ─── Public API ───────────────────────────────────────────────────────────────
def analyze(clean_text: str, semantic_meta: dict) -> dict:
    """
    Parameters
    ----------
    clean_text    : structurally preprocessed text (Track B output)
    semantic_meta : semantic metadata dict (Track A output)
    """
    sem_prob, sem_factors = _semantic_score(semantic_meta)

    nlp_prob = None
    engine   = "semantic_heuristic"

    # Try RoBERTa first
    if clean_text.strip():
        try:
            nlp_prob = _roberta_score(clean_text)
            engine   = "roberta+semantic"
        except Exception as exc:
            logger.warning(f"RoBERTa failed: {exc} — trying TF-IDF")
            try:
                pl       = _load_tfidf()
                if pl:
                    nlp_prob = float(pl.predict_proba([clean_text])[0][1])
                    engine   = "tfidf+semantic"
            except Exception as exc2:
                logger.warning(f"TF-IDF also failed: {exc2}")

    # Fusion: NLP 55% + Semantic metadata 45%
    if nlp_prob is not None:
        final_prob   = 0.55 * nlp_prob + 0.45 * sem_prob
        risk_factors = [f"nlp_score={nlp_prob:.4f}",
                        f"semantic_score={sem_prob:.4f}"] + sem_factors
    else:
        final_prob   = sem_prob
        risk_factors = sem_factors

    final_prob = float(np.clip(final_prob, 0.0, 1.0))

    result = {
        "risk_probability": round(final_prob, 6),
        "risk_factors":     risk_factors[:15],  # top 15 for readability
        "timestamp":        datetime.utcnow().isoformat(),
        "engine":           engine,
    }
    logger.info(json.dumps(result))
    return result

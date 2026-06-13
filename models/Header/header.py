#!/usr/bin/env python3
"""
~/MGW/models/Header/header.py
Header risk analyser — XGBoost (structural fields) + TF-IDF NLP (subject).

Per spec:
  • Subject field → TF-IDF NLP model
  • From / To / Return-Path / Date / Auth → XGBoost ML model
  • Final score = weighted fusion of both sub-scores.
"""
from __future__ import annotations
import os, sys, json, logging, subprocess, math, pickle
from datetime import datetime
from pathlib import Path

MGW_ROOT    = Path.home() / "MGW"
MODEL_DIR   = MGW_ROOT / "models" / "Header"
MODEL_FILE  = MODEL_DIR / "xgb_header_model.json"
SUBJ_TFIDF  = MODEL_DIR / "tfidf_subject_header.pkl"
LOG_FILE    = MODEL_DIR / "header.log"
DATASET_DIR = Path.home() / "Datasets"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("header_analyzer")
if not logger.handlers:
    h = logging.FileHandler(LOG_FILE)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(h)


def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])


for _pkg, _mod in [("numpy","numpy"), ("pandas","pandas"),
                   ("scikit-learn","sklearn"), ("xgboost","xgboost")]:
    try:
        __import__(_mod)
    except ImportError:
        _pip(_pkg)

import numpy  as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

# ─── XGBoost feature columns ──────────────────────────────────────────────────
HEADER_FEATURE_COLS = [
    # Auth signals
    "has_dkim","spf_fail","dkim_fail","dmarc_fail",
    "domain_mismatch","suspicious_tld_sender","has_numeric_in_domain",
    "has_reply_to","has_return_path",
    # Header structure
    "has_from","has_to","has_cc","has_bcc","has_subject",
    "has_date","has_message_id","received_hops","date_is_future",
    # Subject structural signals (NOT NLP — just numeric)
    "subject_all_caps","subject_caps_ratio","subject_money",
    "subject_exclamation","subject_has_numbers","subject_has_special",
    "subject_length","subject_word_count",
    # Financial
    "dollar_count","total_money_symbols",
    # Embedded code (header-visible)
    "has_script","has_iframe","has_form",
    # URL summary
    "url_count","url_has_ip","url_suspicious_tlds","url_mismatch_count",
]

# ─── Heuristic fallback weights ───────────────────────────────────────────────
HEURISTIC_WEIGHTS = {
    "spf_fail":               0.40,
    "dkim_fail":              0.35,
    "dmarc_fail":             0.45,
    "domain_mismatch":        0.40,
    "suspicious_tld_sender":  0.35,
    "date_is_future":         0.30,
    "subject_all_caps":       0.20,
    "subject_money":          0.25,
    "subject_caps_ratio":     0.30,
    "subject_exclamation":    0.10,
    "has_numeric_in_domain":  0.20,
    "url_has_ip":             0.30,
    "url_mismatch_count":     0.35,
    "has_iframe":             0.25,
    "has_script":             0.20,
    "has_dkim":              -0.35,
    "has_message_id":        -0.10,
    "has_return_path":       -0.10,
    "has_date":              -0.05,
}

# ─── Score fusion weights ─────────────────────────────────────────────────────
W_XGB_STRUCT = 0.65     # XGBoost (structural + auth features)
W_NLP_SUBJ   = 0.35     # TF-IDF subject NLP score

_MODEL        = None
_SUBJ_TFIDF_P = None


# ─── XGBoost model ────────────────────────────────────────────────────────────
def _load_or_train_xgb():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if MODEL_FILE.exists():
        clf = xgb.XGBClassifier(use_label_encoder=False,
                                 eval_metric="logloss", verbosity=0)
        clf.load_model(str(MODEL_FILE))
        logger.info("XGBoost header model loaded from disk")
        _MODEL = clf
        return clf

    # Train from master dataset
    master = DATASET_DIR / "master_phishing_dataset.csv"
    csvs   = [master] if master.exists() else list(DATASET_DIR.glob("*.csv"))
    if not csvs:
        logger.warning("No datasets found — heuristic-only mode")
        return None

    dfs = []
    for c in csvs:
        try:
           df_temp = pd.read_csv(c, engine='python', on_bad_lines='skip', encoding_errors='replace')
            if df_temp is not None and len(df_temp) > 0:
            dfs.append(df_temp)
        except Exception as e:
            logger.warning(f"Cannot read {c.name}: {e}")
    if not dfs:
        return None

    df   = pd.concat(dfs, ignore_index=True)
    lcol = next((c for c in ["label","Label","spam","class"] if c in df.columns), None)
    if lcol is None:
        return None
    if lcol != "label":
        df = df.rename(columns={lcol: "label"})

    cols = [c for c in HEADER_FEATURE_COLS if c in df.columns]
    if len(cols) < 4:
        logger.warning(f"Only {len(cols)} header cols in dataset — heuristic only")
        return None

    X = df[cols].fillna(0).astype(float)
    y = df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg / max(pos, 1),
        use_label_encoder=False, eval_metric="auc",
        verbosity=0, random_state=42,
        early_stopping_rounds=20,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_prob = clf.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    logger.info(f"XGBoost trained | ROC-AUC={auc:.4f} | features={cols}")
    clf.save_model(str(MODEL_FILE))
    _MODEL = clf
    return clf


# ─── Subject TF-IDF model ─────────────────────────────────────────────────────
def _load_or_train_subject_tfidf():
    global _SUBJ_TFIDF_P
    if _SUBJ_TFIDF_P is not None:
        return _SUBJ_TFIDF_P

    if SUBJ_TFIDF.exists():
        with open(SUBJ_TFIDF, "rb") as f:
            _SUBJ_TFIDF_P = pickle.load(f)
        logger.info("Subject TF-IDF loaded from disk")
        return _SUBJ_TFIDF_P

    master = DATASET_DIR / "master_phishing_dataset.csv"
    if not master.exists():
        return None

    df = pd.read_csv(master, low_memory=False, engine='python', on_bad_lines='skip', encoding_errors='replace')
    if "label" not in df.columns:
        return None
    subj_col = next((c for c in ["subject","Subject"] if c in df.columns), None)
    if subj_col is None:
        return None

    df = df[[subj_col,"label"]].dropna()
    X  = df[subj_col].astype(str).tolist()
    y  = df["label"].astype(int).tolist()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.1,
                                         random_state=42, stratify=y)
    pl = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                   stop_words="english", sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=1.0,
                                      class_weight="balanced",
                                      solver="lbfgs", random_state=42)),
    ])
    pl.fit(X_tr, y_tr)
    with open(SUBJ_TFIDF, "wb") as f:
        pickle.dump(pl, f)
    logger.info("Subject TF-IDF trained and saved")
    _SUBJ_TFIDF_P = pl
    return pl


def _subject_nlp_score(subject_text: str) -> float | None:
    if not subject_text or not subject_text.strip():
        return None
    try:
        pl = _load_or_train_subject_tfidf()
        if pl:
            return float(pl.predict_proba([subject_text])[0][1])
    except Exception as exc:
        logger.warning(f"Subject TF-IDF inference failed: {exc}")
    return None


# ─── Heuristic fallback ───────────────────────────────────────────────────────
def _heuristic(features: dict):
    positive = negative = 0.0
    factors  = []
    for feat, weight in HEURISTIC_WEIGHTS.items():
        val = float(features.get(feat, 0) or 0)
        if not val:
            continue
        c = weight * val
        if c > 0: positive += c
        else:     negative += abs(c)
        factors.append(f"{feat}={val:.2f} w={weight:+.2f}")
    net  = positive - negative
    prob = (math.tanh(net * 1.2) + 1) / 2
    return float(np.clip(prob, 0.0, 1.0)), factors


# ─── Public API ───────────────────────────────────────────────────────────────
def analyze(header_features: dict) -> dict:
    """
    Parameters
    ----------
    header_features : semantic_meta dict from mail_filter.py (Track A output).
                      Must include 'subject_text' key for NLP scoring.
    """
    # ── XGBoost (structural + auth) ───────────────────────────────────────────
    model        = _load_or_train_xgb()
    risk_factors = []

    if model is not None:
        cols = list(getattr(model, "feature_names_in_",
                            [c for c in HEADER_FEATURE_COLS]))
        row  = {c: float(header_features.get(c, 0) or 0) for c in cols}
        X    = pd.DataFrame([row])
        xgb_prob = float(model.predict_proba(X)[0][1])

        if hasattr(model, "feature_importances_"):
            imps = dict(zip(cols, model.feature_importances_))
            for f, imp in sorted(imps.items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
                risk_factors.append(
                    f"{f}={row.get(f,0):.3f} imp={imp:.4f}")
        engine = "xgboost"
    else:
        xgb_prob, risk_factors = _heuristic(header_features)
        engine = "heuristic"

    # ── Subject TF-IDF NLP ────────────────────────────────────────────────────
    subject_text = header_features.get("subject_text", "")
    subj_prob    = _subject_nlp_score(subject_text)

    # ── Fusion ────────────────────────────────────────────────────────────────
    if subj_prob is not None:
        final_prob = W_XGB_STRUCT * xgb_prob + W_NLP_SUBJ * subj_prob
        risk_factors.insert(0, f"xgb_struct={xgb_prob:.4f}")
        risk_factors.insert(1, f"subject_tfidf={subj_prob:.4f}")
        engine = f"{engine}+subject_tfidf"
    else:
        final_prob = xgb_prob

    final_prob = float(np.clip(final_prob, 0.0, 1.0))

    result = {
        "risk_probability": round(final_prob, 6),
        "risk_factors":     risk_factors,
        "timestamp":        datetime.utcnow().isoformat(),
        "engine":           engine,
    }
    logger.info(json.dumps(result))
    return result

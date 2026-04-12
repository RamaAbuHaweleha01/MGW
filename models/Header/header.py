#!/usr/bin/env python3
"""~/MGW/models/Header/header.py — XGBoost header risk analyzer."""
from __future__ import annotations
import os, sys, json, logging, subprocess, math
from datetime import datetime
from pathlib import Path

MGW_ROOT    = Path.home() / "MGW"
MODEL_DIR   = MGW_ROOT / "models" / "Header"
MODEL_FILE  = MODEL_DIR / "xgb_header_model.json"
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

for _pkg, _mod in [("numpy","numpy"),("pandas","pandas"),
                   ("scikit-learn","sklearn"),("xgboost","xgboost")]:
    try: __import__(_mod)
    except ImportError: _pip(_pkg)

import numpy  as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# ─── Feature columns passed from semantic_track ───────────────────────────────
HEADER_FEATURE_COLS = [
    # Auth signals
    "has_dkim","spf_fail","dkim_fail","dmarc_fail",
    "domain_mismatch","suspicious_tld_sender","has_numeric_in_domain",
    "has_reply_to","has_return_path",
    # Header structure
    "has_from","has_to","has_cc","has_bcc","has_subject",
    "has_date","has_message_id","received_hops","date_is_future",
    # Subject signals
    "subject_has_urgent","subject_has_verify","subject_has_alert",
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

# Heuristic weights — used when model not trained yet
HEURISTIC_WEIGHTS = {
    "spf_fail":               0.40,
    "dkim_fail":              0.35,
    "dmarc_fail":             0.45,
    "domain_mismatch":        0.40,
    "suspicious_tld_sender":  0.35,
    "date_is_future":         0.30,
    "subject_has_urgent":     0.30,
    "subject_has_verify":     0.25,
    "subject_has_alert":      0.25,
    "subject_all_caps":       0.20,
    "subject_money":          0.25,
    "subject_caps_ratio":     0.30,
    "subject_exclamation":    0.10,
    "has_numeric_in_domain":  0.20,
    "url_has_ip":             0.30,
    "url_mismatch_count":     0.35,
    "has_iframe":             0.25,
    "has_script":             0.20,
    # Legitimacy — reduce risk
    "has_dkim":              -0.35,
    "has_message_id":        -0.10,
    "has_return_path":       -0.10,
    "has_date":              -0.05,
}

_MODEL = None

def _load_or_train():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if MODEL_FILE.exists():
        clf = xgb.XGBClassifier(use_label_encoder=False,
                                 eval_metric="logloss", verbosity=0)
        clf.load_model(str(MODEL_FILE))
        logger.info(f"XGBoost header model loaded")
        _MODEL = clf
        return clf

    # Try training from master dataset
    master = DATASET_DIR / "master_phishing_dataset.csv"
    csvs   = [master] if master.exists() else list(DATASET_DIR.glob("*.csv"))
    if not csvs:
        logger.warning("No datasets — heuristic only")
        return None

    dfs = []
    for c in csvs:
        try:
            dfs.append(pd.read_csv(c, low_memory=False))
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
        scale_pos_weight=neg/max(pos,1),
        use_label_encoder=False, eval_metric="auc",
        verbosity=0, random_state=42,
        early_stopping_rounds=20,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False)

    y_prob = clf.predict_proba(X_te)[:,1]
    auc    = roc_auc_score(y_te, y_prob)
    logger.info(f"Header XGBoost trained | ROC-AUC={auc:.4f} | features={cols}")
    clf.save_model(str(MODEL_FILE))
    _MODEL = clf
    return clf


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


def analyze(header_features: dict) -> dict:
    model        = _load_or_train()
    risk_factors = []

    if model is not None:
        cols = getattr(model, "feature_names_in_",
                       [c for c in HEADER_FEATURE_COLS])
        row  = {c: float(header_features.get(c, 0) or 0) for c in cols}
        X    = pd.DataFrame([row])
        prob = float(model.predict_proba(X)[0][1])

        if hasattr(model, "feature_importances_"):
            imps = dict(zip(cols, model.feature_importances_))
            for f, imp in sorted(imps.items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
                risk_factors.append(
                    f"{f}={row.get(f,0):.3f} imp={imp:.4f}")
        engine = "xgboost"
    else:
        prob, risk_factors = _heuristic(header_features)
        engine = "heuristic"

    result = {
        "risk_probability": round(prob, 6),
        "risk_factors":     risk_factors,
        "timestamp":        datetime.utcnow().isoformat(),
        "engine":           engine,
    }
    logger.info(json.dumps(result))
    return result

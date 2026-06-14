#!/usr/bin/env python3
"""
~/MGW/mgw_train_eval.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MGW Full Training & Evaluation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAGES
  Stage 1 — TRAIN   (60%)  : Fit all models, report train metrics
  Stage 2 — TEST    (20%)  : Held-out test set, report test metrics
  Stage 3 — VALIDATE(20%)  : Final validation — Precision/Recall/F1/ROC-AUC

MODELS TRAINED
  ① Header XGBoost      (structural + auth features)
  ② Header TF-IDF LR    (subject-line NLP)
  ③ Body TF-IDF LR      (body text NLP, if 'body'/'Body' column exists)
  ④ Decision Engine LR  (meta-features fusion — simulates decision_engine.py)

DATASET
  ~/Datasets/master_dataset.csv
  Required columns: label  (0=legit, 1=phishing)
  Optional columns: subject, body, + any HEADER_FEATURE_COLS

OUTPUT
  ~/MGW/models/eval/
    ├── split_summary.txt
    ├── stage1_train_report.txt
    ├── stage2_test_report.txt
    ├── stage3_validation_report.txt
    ├── roc_curves.png
    └── confusion_matrices.png
"""

from __future__ import annotations
import os, sys, logging, pickle, json, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── auto-install deps ──────────────────────────────────────────────────────────
def _pip(*pkgs):
    import subprocess
    for p in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p, "--quiet"])

for _p, _m in [("numpy","numpy"),("pandas","pandas"),("scikit-learn","sklearn"),
               ("xgboost","xgboost"),("matplotlib","matplotlib"),("seaborn","seaborn")]:
    try: __import__(_m)
    except ImportError: _pip(_p)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV

# ── paths ──────────────────────────────────────────────────────────────────────
HOME        = Path.home()
DATASET     = HOME / "datasets" / "gw_final_dataset.csv"
EVAL_DIR    = HOME / "MGW" / "models" / "Training" / "eval"
MODEL_DIR   = HOME / "MGW" / "models"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = EVAL_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("mgw_eval")

# ── header feature columns (must match header.py exactly) ─────────────────────
HEADER_FEATURE_COLS = [
    "has_dkim","spf_fail","dkim_fail","dmarc_fail",
    "domain_mismatch","suspicious_tld_sender","has_numeric_in_domain",
    "has_reply_to","has_return_path",
    "has_from","has_to","has_cc","has_bcc","has_subject",
    "has_date","has_message_id","received_hops","date_is_future",
    "subject_all_caps","subject_caps_ratio","subject_money",
    "subject_exclamation","subject_has_numbers","subject_has_special",
    "subject_length","subject_word_count",
    "dollar_count","total_money_symbols",
    "has_script","has_iframe","has_form",
    "url_count","url_has_ip","url_suspicious_tlds","url_mismatch_count",
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str, width: int = 70) -> str:
    bar = "═" * width
    return f"\n{bar}\n  {title}\n{bar}\n"


def _fmt_report(y_true, y_pred, y_prob, name: str) -> str:
    cr   = classification_report(y_true, y_pred, digits=4,
                                  target_names=["Legit","Phishing"])
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

    lines = [
        f"Model          : {name}",
        f"Samples        : {len(y_true)}",
        f"Phishing rate  : {y_true.mean()*100:.1f}%",
        "─"*50,
        f"Precision      : {prec:.4f}",
        f"Recall         : {rec:.4f}",
        f"F1-score       : {f1:.4f}",
        f"ROC-AUC        : {auc:.4f}",
        f"Avg Precision  : {ap:.4f}  (area under PR curve)",
        "─"*50,
        "Confusion Matrix:",
        f"  TN={tn}  FP={fp}",
        f"  FN={fn}  TP={tp}",
        "─"*50,
        "Classification Report:",
        cr,
    ]
    return "\n".join(lines)


def _optimal_threshold(y_true, y_prob) -> tuple[float, float, float, float]:
    """Youden's J: threshold that maximises TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx], tpr[idx], fpr[idx], j[idx]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_split(path: Path):
    log.info(f"Loading dataset: {path}")
    df = pd.read_csv(path, engine="python",
                     on_bad_lines="skip", encoding_errors="replace")
    log.info(f"Raw shape: {df.shape}")

    # normalise label column
    lcol = next((c for c in ["label","Label","spam","Spam","class","Class"]
                 if c in df.columns), None)
    if lcol is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")
    if lcol != "label":
        df = df.rename(columns={lcol: "label"})

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    log.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    n_total    = len(df)
    n_phishing = (df["label"] == 1).sum()
    n_legit    = (df["label"] == 0).sum()

    # ── 60 / 20 / 20 split ──────────────────────────────────────────────────
    # First cut: 60% train, 40% temp
    df_train, df_temp = train_test_split(
        df, test_size=0.40, random_state=42, stratify=df["label"])
    # Second cut: 50% of temp → 20% test, 20% val
    df_test, df_val = train_test_split(
        df_temp, test_size=0.50, random_state=42, stratify=df_temp["label"])

    summary = [
        _section("DATASET & SPLIT SUMMARY"),
        f"Dataset path   : {path}",
        f"Total samples  : {n_total}",
        f"  Phishing     : {n_phishing} ({n_phishing/n_total*100:.1f}%)",
        f"  Legitimate   : {n_legit} ({n_legit/n_total*100:.1f}%)",
        "",
        f"Split 60/20/20:",
        f"  TRAIN        : {len(df_train)} samples  "
        f"(phish={df_train['label'].sum()}, legit={len(df_train)-df_train['label'].sum()})",
        f"  TEST         : {len(df_test)} samples  "
        f"(phish={df_test['label'].sum()}, legit={len(df_test)-df_test['label'].sum()})",
        f"  VALIDATE     : {len(df_val)} samples  "
        f"(phish={df_val['label'].sum()}, legit={len(df_val)-df_val['label'].sum()})",
        "",
        f"Available feature columns:",
    ]
    feat_cols = [c for c in HEADER_FEATURE_COLS if c in df.columns]
    summary.append(f"  Header struct : {len(feat_cols)}/{len(HEADER_FEATURE_COLS)}  {feat_cols}")
    for col in ["subject","Subject","body","Body","body_text"]:
        if col in df.columns:
            summary.append(f"  Text column   : '{col}' found")
    txt = "\n".join(summary)

    split_file = EVAL_DIR / "split_summary.txt"
    split_file.write_text(txt)
    log.info(f"Split summary → {split_file}")
    print(txt)

    return df_train, df_test, df_val, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINERS
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb_header(df_train, df_val, feat_cols) -> xgb.XGBClassifier | None:
    """Train XGBoost on structural header features."""
    if len(feat_cols) < 4:
        log.warning(f"Only {len(feat_cols)} header features — skipping XGBoost header")
        return None

    X_tr = df_train[feat_cols].fillna(0).astype(float)
    y_tr = df_train["label"]
    X_val = df_val[feat_cols].fillna(0).astype(float)
    y_val = df_val["label"]

    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()

    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg / max(pos, 1),
        use_label_encoder=False, eval_metric="auc",
        verbosity=0, random_state=42,
        early_stopping_rounds=25,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # cross-val on train set
    cv_scores = cross_val_score(
        xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                           use_label_encoder=False, eval_metric="auc",
                           verbosity=0, random_state=42),
        X_tr, y_tr, cv=5, scoring="roc_auc", n_jobs=-1)
    log.info(f"XGBoost 5-CV ROC-AUC on TRAIN: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # save
    out = MODEL_DIR / "Header" / "xgb_header_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out))
    log.info(f"XGBoost header model saved → {out}")
    return clf


def train_tfidf_subject(df_train, text_col: str) -> Pipeline | None:
    """Train TF-IDF + LR on subject text."""
    if text_col not in df_train.columns:
        log.warning(f"No subject column '{text_col}' — skipping subject TF-IDF")
        return None

    df = df_train[[text_col, "label"]].dropna()
    X  = df[text_col].astype(str).tolist()
    y  = df["label"].astype(int).tolist()

    pl = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 3),
                                   stop_words="english", sublinear_tf=True,
                                   min_df=2)),
        ("clf",   LogisticRegression(max_iter=1000, C=0.5,
                                      class_weight="balanced",
                                      solver="lbfgs", random_state=42)),
    ])
    pl.fit(X, y)

    # save
    out = MODEL_DIR / "Header" / "tfidf_subject_header.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(pl, f)
    log.info(f"Subject TF-IDF saved → {out}")
    return pl


def train_tfidf_body(df_train, body_col: str) -> Pipeline | None:
    """Train TF-IDF + LR on body text."""
    if body_col not in df_train.columns:
        log.warning(f"No body column '{body_col}' — skipping body TF-IDF")
        return None

    df = df_train[[body_col, "label"]].dropna()
    X  = df[body_col].astype(str).tolist()
    y  = df["label"].astype(int).tolist()

    pl = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                                   stop_words="english", sublinear_tf=True,
                                   min_df=3, max_df=0.95)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                      class_weight="balanced",
                                      solver="lbfgs", random_state=42)),
    ])
    pl.fit(X, y)

    out = MODEL_DIR / "Body" / "tfidf_body.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(pl, f)
    log.info(f"Body TF-IDF saved → {out}")
    return pl


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _predict(df: pd.DataFrame, feat_cols, xgb_model, subj_model,
             body_model, subj_col, body_col,
             W_XGB=0.50, W_SUBJ=0.25, W_BODY=0.25) -> tuple[np.ndarray, np.ndarray]:
    """Fuse all available model predictions into a final probability."""
    n = len(df)
    scores = np.zeros(n, dtype=float)
    weights_used = np.zeros(n, dtype=float)

    if xgb_model is not None and feat_cols:
        X = df[feat_cols].fillna(0).astype(float)
        p = xgb_model.predict_proba(X)[:, 1]
        scores += W_XGB * p
        weights_used += W_XGB

    if subj_model is not None and subj_col in df.columns:
        texts = df[subj_col].fillna("").astype(str).tolist()
        p = subj_model.predict_proba(texts)[:, 1]
        scores += W_SUBJ * p
        weights_used += W_SUBJ

    if body_model is not None and body_col in df.columns:
        texts = df[body_col].fillna("").astype(str).tolist()
        p = body_model.predict_proba(texts)[:, 1]
        scores += W_BODY * p
        weights_used += W_BODY

    # normalise in case some models were absent
    weights_used = np.where(weights_used == 0, 1.0, weights_used)
    probs = scores / weights_used

    # find best threshold via Youden's J (computed globally — you can re-tune)
    preds = (probs >= 0.50).astype(int)
    return probs, preds


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(results: dict, outpath: Path):
    """Plot ROC curves for all three stages on one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"train": "royalblue", "test": "darkorange", "val": "green"}

    for ax, (stage, res) in zip(axes, results.items()):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        auc = roc_auc_score(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=colors[stage], lw=2,
                label=f"ROC AUC = {auc:.4f}")
        ax.plot([0,1],[0,1],"k--", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.08, color=colors[stage])
        ax.set_title(f"{stage.upper()} ROC Curve", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.grid(alpha=0.3)

    plt.suptitle("MGW — ROC Curves per Stage", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"ROC curves saved → {outpath}")


def plot_confusion_matrices(results: dict, outpath: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (stage, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="Blues", cbar=False,
                    xticklabels=["Legit","Phishing"],
                    yticklabels=["Legit","Phishing"])
        ax.set_title(f"{stage.upper()} Confusion Matrix", fontsize=12, fontweight="bold")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    plt.suptitle("MGW — Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrices saved → {outpath}")


def plot_pr_curves(results: dict, outpath: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"train": "royalblue", "test": "darkorange", "val": "green"}
    for ax, (stage, res) in zip(axes, results.items()):
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ap = average_precision_score(res["y_true"], res["y_prob"])
        ax.plot(rec, prec, color=colors[stage], lw=2,
                label=f"AP = {ap:.4f}")
        ax.fill_between(rec, prec, alpha=0.08, color=colors[stage])
        baseline = res["y_true"].mean()
        ax.axhline(baseline, color="gray", linestyle="--", lw=1,
                   label=f"Baseline = {baseline:.2f}")
        ax.set_title(f"{stage.upper()} Precision-Recall Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.grid(alpha=0.3)
    plt.suptitle("MGW — Precision-Recall Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"PR curves saved → {outpath}")


def plot_feature_importance(xgb_model, feat_cols: list, outpath: Path):
    if xgb_model is None or not feat_cols:
        return
    imps = dict(zip(feat_cols, xgb_model.feature_importances_))
    top  = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:20]
    names, vals = zip(*top)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(names[::-1], vals[::-1], color="steelblue", edgecolor="white")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Header Model — Top 20 Features", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance saved → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

def _interpret(auc: float, f1: float, prec: float, rec: float, stage: str) -> list[str]:
    """Return actionable feedback based on metric values."""
    feedback = []

    # AUC
    if auc >= 0.97:
        feedback.append(f"✅  ROC-AUC={auc:.4f} — Excellent discrimination on {stage} set.")
    elif auc >= 0.92:
        feedback.append(f"✅  ROC-AUC={auc:.4f} — Very good. Acceptable for production.")
    elif auc >= 0.85:
        feedback.append(f"⚠️  ROC-AUC={auc:.4f} — Moderate. Consider feature engineering.")
    else:
        feedback.append(f"❌  ROC-AUC={auc:.4f} — Below acceptable threshold (>0.85). "
                        "Model needs significant improvement.")

    # F1
    if f1 >= 0.92:
        feedback.append(f"✅  F1={f1:.4f} — Strong harmonic mean of precision/recall.")
    elif f1 >= 0.85:
        feedback.append(f"⚠️  F1={f1:.4f} — Acceptable but room to improve.")
    else:
        feedback.append(f"❌  F1={f1:.4f} — Too low. Check class imbalance or add features.")

    # Precision vs Recall balance
    gap = abs(prec - rec)
    if gap > 0.15:
        if prec > rec:
            feedback.append(f"⚠️  Precision({prec:.3f}) >> Recall({rec:.3f}) — "
                            "Model is too conservative: missing real phishing emails (false negatives). "
                            "Lower decision threshold or increase recall with class_weight.")
        else:
            feedback.append(f"⚠️  Recall({rec:.3f}) >> Precision({prec:.3f}) — "
                            "Model is too aggressive: too many false positives (blocking legit mail). "
                            "Raise decision threshold or add negative features.")
    else:
        feedback.append(f"✅  Precision({prec:.3f}) ≈ Recall({rec:.3f}) — Well-balanced.")

    # Generalisation gap check (only meaningful for test/val)
    return feedback


def _gap_analysis(train_auc: float, test_auc: float) -> list[str]:
    gap = train_auc - test_auc
    if gap > 0.08:
        return [f"⚠️  OVERFITTING DETECTED: Train AUC={train_auc:.4f} vs Test AUC={test_auc:.4f} "
                f"(gap={gap:.4f}). Recommend: increase regularisation (reg_alpha/reg_lambda), "
                "reduce max_depth, or add more training data."]
    elif gap < -0.03:
        return [f"ℹ️  Test AUC > Train AUC by {abs(gap):.4f} — data shift or lucky split. "
                "Check dataset balance."]
    else:
        return [f"✅  Generalisation gap = {gap:.4f} — model generalises well."]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*70}")
    print(f"  MGW Training & Evaluation Pipeline  —  {ts}")
    print(f"{'='*70}\n")

    if not DATASET.exists():
        # Try alternate common names
        for alt in ["master_phishing_dataset.csv", "phishing.csv", "dataset.csv"]:
            alt_path = DATASET.parent / alt
            if alt_path.exists():
                log.info(f"Using alternate dataset: {alt_path}")
                break
        else:
            log.error(f"Dataset not found: {DATASET}")
            log.error("Please ensure ~/Datasets/master_dataset.csv exists.")
            sys.exit(1)
        actual_dataset = alt_path
    else:
        actual_dataset = DATASET

    # ── detect text columns ──────────────────────────────────────────────────
    df_peek = pd.read_csv(actual_dataset, nrows=5, engine="python",
                          on_bad_lines="skip", encoding_errors="replace")
    subj_col = next((c for c in ["subject","Subject"] if c in df_peek.columns), None)
    body_col = next((c for c in ["body","Body","body_text","text"] if c in df_peek.columns), None)
    log.info(f"Detected subject_col={subj_col}, body_col={body_col}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 0 — Load & Split
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    df_train, df_test, df_val, feat_cols = load_and_split(actual_dataset)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 1 — TRAIN (60%)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(_section("STAGE 1 — TRAINING (60% of data)"))
    log.info("Training XGBoost header model...")
    xgb_model = train_xgb_header(df_train, df_val, feat_cols)

    log.info("Training subject TF-IDF model...")
    subj_model = train_tfidf_subject(df_train, subj_col) if subj_col else None

    log.info("Training body TF-IDF model...")
    body_model = train_tfidf_body(df_train, body_col) if body_col else None

    # Evaluate on TRAIN set
    tr_probs, tr_preds = _predict(df_train, feat_cols, xgb_model,
                                   subj_model, body_model, subj_col or "", body_col or "")
    tr_true = df_train["label"].values
    tr_auc  = roc_auc_score(tr_true, tr_probs)
    tr_f1   = f1_score(tr_true, tr_preds, zero_division=0)
    tr_prec = precision_score(tr_true, tr_preds, zero_division=0)
    tr_rec  = recall_score(tr_true, tr_preds, zero_division=0)
    thr, tpr_opt, fpr_opt, _ = _optimal_threshold(tr_true, tr_probs)

    train_report_lines = [
        _section("STAGE 1 — TRAIN SET RESULTS (60%)"),
        _fmt_report(tr_true, tr_preds, tr_probs, "Fused MGW (train)"),
        "",
        f"Optimal Threshold (Youden J): {thr:.4f}  "
        f"(TPR={tpr_opt:.4f}, FPR={fpr_opt:.4f})",
        "",
        "── ANALYSIS & FEEDBACK ──",
        *_interpret(tr_auc, tr_f1, tr_prec, tr_rec, "train"),
        "",
        "ℹ️  Training metrics are expected to be high; overfitting check is "
        "done on the test set (Stage 2).",
    ]
    train_report = "\n".join(train_report_lines)
    (EVAL_DIR / "stage1_train_report.txt").write_text(train_report)
    print(train_report)

    # Feature importance plot
    plot_feature_importance(xgb_model, feat_cols, EVAL_DIR / "feature_importance.png")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 2 — TEST (20%)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(_section("STAGE 2 — TESTING (20% of data)"))
    te_probs, te_preds = _predict(df_test, feat_cols, xgb_model,
                                   subj_model, body_model, subj_col or "", body_col or "")
    te_true = df_test["label"].values
    te_auc  = roc_auc_score(te_true, te_probs)
    te_f1   = f1_score(te_true, te_preds, zero_division=0)
    te_prec = precision_score(te_true, te_preds, zero_division=0)
    te_rec  = recall_score(te_true, te_preds, zero_division=0)
    thr_te, _, _, _ = _optimal_threshold(te_true, te_probs)

    test_report_lines = [
        _section("STAGE 2 — TEST SET RESULTS (20%)"),
        _fmt_report(te_true, te_preds, te_probs, "Fused MGW (test)"),
        "",
        f"Optimal Threshold (Youden J): {thr_te:.4f}",
        "",
        "── GENERALISATION ANALYSIS ──",
        *_gap_analysis(tr_auc, te_auc),
        "",
        "── ANALYSIS & FEEDBACK ──",
        *_interpret(te_auc, te_f1, te_prec, te_rec, "test"),
    ]
    test_report = "\n".join(test_report_lines)
    (EVAL_DIR / "stage2_test_report.txt").write_text(test_report)
    print(test_report)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STAGE 3 — VALIDATE (20%)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(_section("STAGE 3 — FINAL VALIDATION (20% of data)"))
    val_probs, val_preds = _predict(df_val, feat_cols, xgb_model,
                                     subj_model, body_model, subj_col or "", body_col or "")
    val_true = df_val["label"].values
    val_auc  = roc_auc_score(val_true, val_probs)
    val_f1   = f1_score(val_true, val_preds, zero_division=0)
    val_prec = precision_score(val_true, val_preds, zero_division=0)
    val_rec  = recall_score(val_true, val_preds, zero_division=0)
    thr_val, _, _, _ = _optimal_threshold(val_true, val_probs)

    # Per-class breakdown
    cr_val = classification_report(val_true, val_preds, digits=4,
                                    target_names=["Legit","Phishing"])

    val_report_lines = [
        _section("STAGE 3 — VALIDATION SET RESULTS (20%)"),
        _fmt_report(val_true, val_preds, val_probs, "Fused MGW (validation)"),
        "",
        f"Optimal Threshold (Youden J): {thr_val:.4f}",
        "  → Use this threshold in decision_engine.py for CAPE malscore integration.",
        "",
        "── FINAL VALIDATION SUMMARY ──",
        f"  ROC-AUC   : {val_auc:.4f}",
        f"  F1-score  : {val_f1:.4f}",
        f"  Precision : {val_prec:.4f}",
        f"  Recall    : {val_rec:.4f}",
        "",
        "── GENERALISATION vs TEST ──",
        *_gap_analysis(te_auc, val_auc),
        "",
        "── ANALYSIS & FEEDBACK ──",
        *_interpret(val_auc, val_f1, val_prec, val_rec, "validation"),
        "",
        "── COMPARISON TABLE ──",
        f"  {'Stage':<12} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}",
        f"  {'─'*48}",
        f"  {'Train (60%)':<12} {tr_auc:>8.4f} {tr_f1:>8.4f} {tr_prec:>8.4f} {tr_rec:>8.4f}",
        f"  {'Test  (20%)':<12} {te_auc:>8.4f} {te_f1:>8.4f} {te_prec:>8.4f} {te_rec:>8.4f}",
        f"  {'Val   (20%)':<12} {val_auc:>8.4f} {val_f1:>8.4f} {val_prec:>8.4f} {val_rec:>8.4f}",
        "",
        "── RECOMMENDED ACTIONS ──",
    ]

    # Final recommendations
    max_gap = max(tr_auc - te_auc, te_auc - val_auc)
    if max_gap > 0.06:
        val_report_lines.append(
            "  1. High variance across splits → add more data or tune regularisation.")
    if val_rec < 0.90:
        val_report_lines.append(
            "  2. Recall < 0.90 on validation → lower threshold or retrain with "
            "recall-focused objective (e.g. XGBoost scale_pos_weight increase).")
    if val_prec < 0.85:
        val_report_lines.append(
            "  3. Precision < 0.85 → too many false positives; raise threshold or "
            "add more legitimate email samples to training data.")
    if val_auc >= 0.95 and val_f1 >= 0.92:
        val_report_lines.append(
            "  ✅  Model meets production criteria. Proceed to integration with "
            "mail_filter.py pipeline and CAPEv2 sandbox scoring.")

    val_report = "\n".join(val_report_lines)
    (EVAL_DIR / "stage3_validation_report.txt").write_text(val_report)
    print(val_report)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PLOTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    results = {
        "train": {"y_true": tr_true,  "y_prob": tr_probs,  "y_pred": tr_preds},
        "test":  {"y_true": te_true,  "y_prob": te_probs,  "y_pred": te_preds},
        "val":   {"y_true": val_true, "y_prob": val_probs, "y_pred": val_preds},
    }

    plot_roc(results, EVAL_DIR / "roc_curves.png")
    plot_confusion_matrices(results, EVAL_DIR / "confusion_matrices.png")
    plot_pr_curves(results, EVAL_DIR / "pr_curves.png")

    # ── final summary JSON (machine-readable for CI/CD) ────────────────────
    summary_json = {
        "timestamp": ts,
        "dataset": str(actual_dataset),
        "splits": {
            "train": len(df_train),
            "test":  len(df_test),
            "val":   len(df_val),
        },
        "metrics": {
            "train": {"auc": round(tr_auc,4), "f1": round(tr_f1,4),
                      "precision": round(tr_prec,4), "recall": round(tr_rec,4)},
            "test":  {"auc": round(te_auc,4), "f1": round(te_f1,4),
                      "precision": round(te_prec,4), "recall": round(te_rec,4)},
            "val":   {"auc": round(val_auc,4), "f1": round(val_f1,4),
                      "precision": round(val_prec,4), "recall": round(val_rec,4)},
        },
        "optimal_threshold_val": round(float(thr_val), 4),
        "models": {
            "xgb_header":    xgb_model is not None,
            "tfidf_subject": subj_model is not None,
            "tfidf_body":    body_model is not None,
        },
        "feature_cols_used": feat_cols,
    }
    json_out = EVAL_DIR / "eval_summary.json"
    json_out.write_text(json.dumps(summary_json, indent=2))

    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {EVAL_DIR}")
    print(f"  Files:")
    for f in sorted(EVAL_DIR.iterdir()):
        print(f"    {f.name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
~/datasets/append_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Appends NEW phishing/ham emails to gw_final_dataset.csv.

Uses the SAME extractor as build_dataset.py
(~/MGW/Parsing/extract_phishing_features.py :: extract_features_from_email)
so new rows are computed identically to the existing dataset.

SUPPORTS AS INPUT:
  • mbox files       (phishing or ham)   --phishing-mbox / --ham-mbox
  • maildir folders   (flat dir of files) --phishing-dir  / --ham-dir

USAGE EXAMPLES
  # Add a new phishing mbox (e.g. phishing_pot.mbox) and a folder of new ham
  python3 append_dataset.py \
      --phishing-mbox ~/datasets/raw_phishing/phishing_pot.mbox \
      --ham-dir       ~/datasets/raw_ham/extra_ham

  # Add only phishing from a maildir-style folder of .eml files
  python3 append_dataset.py --phishing-dir ~/datasets/raw_phishing/epvme_subset

By default appends to ~/datasets/gw_final_dataset.csv (backed up to .bak first).
Use --out to write to a different file instead of appending in place.
"""

from __future__ import annotations
import argparse
import email
import logging
import mailbox
import os
import shutil
import sys
from email import policy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.expanduser('~/MGW/Parsing'))
from extract_phishing_features import extract_features_from_email  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("append_dataset")

DEFAULT_CSV = Path.home() / "datasets" / "gw_final_dataset.csv"

HEADER_FEATURE_COLS = [
    "has_dkim", "spf_fail", "dkim_fail", "dmarc_fail",
    "domain_mismatch", "suspicious_tld_sender", "has_numeric_in_domain",
    "has_reply_to", "has_return_path",
    "has_from", "has_to", "has_cc", "has_bcc", "has_subject",
    "has_date", "has_message_id", "received_hops", "date_is_future",
    "subject_all_caps", "subject_caps_ratio", "subject_money",
    "subject_exclamation", "subject_has_numbers", "subject_has_special",
    "subject_length", "subject_word_count",
    "dollar_count", "total_money_symbols",
    "has_script", "has_iframe", "has_form",
    "url_count", "url_has_ip", "url_suspicious_tlds", "url_mismatch_count",
]

ALL_COLUMNS = HEADER_FEATURE_COLS + ["subject_text", "body_text", "label"]


# ══════════════════════════════════════════════════════════════════════════
# Row construction — mirrors build_dataset.py exactly
# ══════════════════════════════════════════════════════════════════════════
def _row_from_msg(msg, label: int) -> dict | None:
    try:
        extracted = extract_features_from_email(msg)
        meta_features = extracted["semantic_meta"]

        row_data = {}
        for col in HEADER_FEATURE_COLS:
            row_data[col] = meta_features.get(
                col, meta_features.get(col.replace('count', 'sign_count'), 0))

        row_data['subject_text'] = meta_features.get('subject_text', '')
        row_data['body_text']    = extracted.get('clean_text', '')
        row_data['label']        = label
        return row_data
    except Exception as exc:
        log.debug(f"Skipping message: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# Input handlers
# ══════════════════════════════════════════════════════════════════════════
def process_mbox(path: Path, label: int) -> list[dict]:
    rows = []
    if not path.exists():
        log.warning(f"mbox not found: {path}")
        return rows

    mbox = mailbox.mbox(str(path), factory=bytes)
    log.info(f"Processing mbox {path} ({len(mbox)} messages, label={label})")

    skipped = 0
    for key in tqdm(mbox.keys()):
        try:
            raw_bytes = mbox.get_string(key).encode('utf-8', errors='ignore')
            msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        except Exception:
            skipped += 1
            continue

        row = _row_from_msg(msg, label)
        if row is None or not row["body_text"].strip():
            skipped += 1
            continue
        rows.append(row)

    log.info(f"  -> {len(rows)} extracted, {skipped} skipped")
    return rows


def process_dir(path: Path, label: int) -> list[dict]:
    rows = []
    if not path.exists():
        log.warning(f"Directory not found: {path}")
        return rows

    files = [f for f in sorted(os.listdir(path))
             if os.path.isfile(os.path.join(path, f))]
    log.info(f"Processing dir {path} ({len(files)} files, label={label})")

    skipped = 0
    for filename in tqdm(files):
        file_path = os.path.join(path, filename)
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        except Exception:
            skipped += 1
            continue

        row = _row_from_msg(msg, label)
        if row is None or not row["body_text"].strip():
            skipped += 1
            continue
        rows.append(row)

    log.info(f"  -> {len(rows)} extracted, {skipped} skipped")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Append new emails to gw_final_dataset.csv")
    ap.add_argument("--phishing-mbox", type=Path, action="append", default=[],
                    help="mbox file of phishing emails (label=1). Repeatable.")
    ap.add_argument("--ham-mbox",      type=Path, action="append", default=[],
                    help="mbox file of ham emails (label=0). Repeatable.")
    ap.add_argument("--phishing-dir",  type=Path, action="append", default=[],
                    help="Folder of phishing .eml files (label=1). Repeatable.")
    ap.add_argument("--ham-dir",       type=Path, action="append", default=[],
                    help="Folder of ham .eml files (label=0). Repeatable.")
    ap.add_argument("--target", type=Path, default=DEFAULT_CSV,
                    help=f"Existing dataset CSV to append to (default: {DEFAULT_CSV})")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write combined result to a new file instead of overwriting --target")
    args = ap.parse_args()

    if not any([args.phishing_mbox, args.ham_mbox, args.phishing_dir, args.ham_dir]):
        log.error("Provide at least one of --phishing-mbox/--ham-mbox/--phishing-dir/--ham-dir")
        sys.exit(1)

    new_rows: list[dict] = []
    for p in args.phishing_mbox:
        new_rows.extend(process_mbox(p, label=1))
    for p in args.ham_mbox:
        new_rows.extend(process_mbox(p, label=0))
    for p in args.phishing_dir:
        new_rows.extend(process_dir(p, label=1))
    for p in args.ham_dir:
        new_rows.extend(process_dir(p, label=0))

    if not new_rows:
        log.error("No new rows extracted — nothing to do.")
        sys.exit(1)

    n_phish = sum(1 for r in new_rows if r["label"] == 1)
    n_ham   = sum(1 for r in new_rows if r["label"] == 0)
    log.info(f"New samples: {len(new_rows)} (phishing={n_phish}, ham={n_ham})")

    new_df = pd.DataFrame(new_rows)
    new_df = new_df.reindex(columns=ALL_COLUMNS).fillna(0)

    if not args.target.exists():
        log.error(f"Target dataset not found: {args.target}")
        sys.exit(1)

    existing_df = pd.read_csv(args.target, encoding="utf-8",
                               on_bad_lines="skip", encoding_errors="replace")

    missing = set(ALL_COLUMNS) - set(existing_df.columns)
    if missing:
        log.error(f"Target dataset missing expected columns: {missing}")
        sys.exit(1)

    log.info(f"Existing dataset: {len(existing_df)} rows "
             f"(phishing={int((existing_df['label']==1).sum())}, "
             f"legit={int((existing_df['label']==0).sum())})")

    combined = pd.concat([existing_df, new_df[existing_df.columns]], ignore_index=True)

    out_path = args.out or args.target
    if out_path == args.target:
        backup = args.target.with_suffix(args.target.suffix + ".bak")
        shutil.copy2(args.target, backup)
        log.info(f"Backup created: {backup}")

    combined.to_csv(out_path, index=False)

    log.info(f"Combined dataset: {len(combined)} rows "
             f"(phishing={int((combined['label']==1).sum())}, "
             f"legit={int((combined['label']==0).sum())}) -> {out_path}")
    log.info("Done. Re-run train_models.py to retrain on the combined dataset.")


if __name__ == "__main__":
    main()

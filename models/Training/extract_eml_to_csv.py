#!/usr/bin/env python3
"""
extract_eml_to_csv.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Converts raw .eml files into rows matching gw_final_dataset.csv schema.

Replicates the EXACT feature extraction logic from
~/MGW/mail_filter.py (semantic_track + extract_body), so new samples
are computed identically to the existing dataset.

USAGE
  python3 extract_eml_to_csv.py \
      --phishing-dir ~/datasets/raw_eml/phishing_eml \
      --legit-dir    ~/datasets/raw_eml/legit_eml \
      --out          ~/datasets/new_samples.csv \
      [--append ~/datasets/gw_final_dataset.csv]

If --append is given, new_samples.csv rows are appended to that file
(after verifying the header matches), producing an updated combined
dataset at <append_path> with a .bak backup of the original.

OUTPUT COLUMNS (must match gw_final_dataset.csv exactly):
  has_dkim,spf_fail,dkim_fail,dmarc_fail,domain_mismatch,
  suspicious_tld_sender,has_numeric_in_domain,has_reply_to,
  has_return_path,has_from,has_to,has_cc,has_bcc,has_subject,
  has_date,has_message_id,received_hops,date_is_future,
  subject_all_caps,subject_caps_ratio,subject_money,
  subject_exclamation,subject_has_numbers,subject_has_special,
  subject_length,subject_word_count,dollar_count,total_money_symbols,
  has_script,has_iframe,has_form,url_count,url_has_ip,
  url_suspicious_tlds,url_mismatch_count,subject_text,body_text,label
"""

from __future__ import annotations
import argparse
import csv
import html
import logging
import re
import shutil
import sys
from datetime import timezone as _tz
import datetime as _dt
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("extract_eml")

# ── Exact column order from gw_final_dataset.csv ───────────────────────────
COLUMNS = [
    "has_dkim", "spf_fail", "dkim_fail", "dmarc_fail", "domain_mismatch",
    "suspicious_tld_sender", "has_numeric_in_domain", "has_reply_to",
    "has_return_path", "has_from", "has_to", "has_cc", "has_bcc",
    "has_subject", "has_date", "has_message_id", "received_hops",
    "date_is_future", "subject_all_caps", "subject_caps_ratio",
    "subject_money", "subject_exclamation", "subject_has_numbers",
    "subject_has_special", "subject_length", "subject_word_count",
    "dollar_count", "total_money_symbols", "has_script", "has_iframe",
    "has_form", "url_count", "url_has_ip", "url_suspicious_tlds",
    "url_mismatch_count", "subject_text", "body_text", "label",
]

# ── Shared constants (copied verbatim from mail_filter.py semantic_track) ──
URL_PATTERN  = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s<>"]*)?', re.I)
HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.I)
MISMATCH_PATTERN = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', re.I)

BAD_TLDS = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
            '.club', '.online', '.site', '.work', '.date', '.loan'}


# ══════════════════════════════════════════════════════════════════════════
# extract_body  (verbatim port of mail_filter.py:extract_body)
# ══════════════════════════════════════════════════════════════════════════
def extract_body(msg) -> str:
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct in ("text/plain", "text/html"):
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            parts.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# semantic_track  (port of mail_filter.py:semantic_track, trimmed to the
# 35 HEADER_FEATURE_COLS + subject_text + body_text needed for training)
# ══════════════════════════════════════════════════════════════════════════
def semantic_features(msg, raw_body: str) -> dict:
    subject  = msg.get("Subject", "") or ""
    text     = raw_body
    lower    = text.lower()
    subj_low = subject.lower()

    all_urls  = URL_PATTERN.findall(text)

    mismatches = sum(
        1 for m in MISMATCH_PATTERN.finditer(text)
        if m.group(2).strip().startswith("http")
        and m.group(2).strip() not in m.group(1)
    )
    ip_in_url = sum(1 for u in all_urls if re.search(r'https?://\d+\.\d+\.\d+\.\d+', u))
    suspicious_tld_count = sum(
        1 for u in all_urls if '/' in u and len(u.split('/')) > 2
        and any(u.lower().split('/')[2].endswith(t) for t in BAD_TLDS))

    has_script = int(bool(re.search(r'<script', text, re.I)))
    has_iframe = int(bool(re.search(r'<iframe', text, re.I)))
    has_form   = int(bool(re.search(r'<form', text, re.I)))

    subject_all_caps    = int(subject.isupper() and len(subject) > 3)
    subject_caps_ratio  = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
    subject_money       = int(any(s in subject for s in ["$", "€", "£", "¥", "money", "wire"]))
    subject_exclamation = subject.count("!")
    subject_has_numbers = int(any(c.isdigit() for c in subject))
    subject_has_special = int(any(c in "!@#$%^&*()" for c in subject))

    auth_results = msg.get("Authentication-Results", "") or ""
    auth_lower   = auth_results.lower()
    spf_fail     = int("spf=fail"  in auth_lower)
    dkim_fail    = int("dkim=fail" in auth_lower)
    dmarc_fail   = int("dmarc=fail" in auth_lower)
    has_dkim     = int(bool(msg.get("DKIM-Signature")))

    from_addr   = msg.get("From", "") or ""
    reply_to    = msg.get("Reply-To", "") or ""
    return_path = msg.get("Return-Path", "") or ""
    from_domain   = from_addr.split("@")[-1].strip(">").strip()   if "@" in from_addr   else ""
    reply_domain  = reply_to.split("@")[-1].strip(">").strip()    if "@" in reply_to    else ""
    return_domain = return_path.strip("<>").split("@")[-1].strip() if "@" in return_path else ""
    domain_mismatch = int(
        (bool(reply_to)    and reply_domain  != from_domain) or
        (bool(return_path) and return_domain != from_domain)
    )
    suspicious_tld_sender = int(any(from_domain.endswith(t) for t in BAD_TLDS))
    has_numeric_in_domain = int(any(c.isdigit() for c in from_domain))
    received_hops = len(msg.get_all("Received", []))

    date_str       = msg.get("Date", "") or ""
    date_is_future = 0
    if date_str:
        try:
            date_obj  = parsedate_to_datetime(date_str)
            now_utc   = _dt.datetime.now(_tz.utc)
            date_utc  = date_obj.astimezone(_tz.utc)
            skew_secs = (date_utc - now_utc).total_seconds()
            date_is_future = int(skew_secs > 300)
        except Exception:
            pass

    dollar_count = text.count("$")
    total_money  = dollar_count + text.count("€") + text.count("£")

    return {
        "has_dkim":               has_dkim,
        "spf_fail":               spf_fail,
        "dkim_fail":              dkim_fail,
        "dmarc_fail":             dmarc_fail,
        "domain_mismatch":        domain_mismatch,
        "suspicious_tld_sender":  suspicious_tld_sender,
        "has_numeric_in_domain":  has_numeric_in_domain,
        "has_reply_to":           int(bool(reply_to)),
        "has_return_path":        int(bool(return_path)),
        "has_from":               int(bool(msg.get("From"))),
        "has_to":                 int(bool(msg.get("To"))),
        "has_cc":                 int(bool(msg.get("Cc"))),
        "has_bcc":                int(bool(msg.get("Bcc"))),
        "has_subject":            int(bool(msg.get("Subject"))),
        "has_date":               int(bool(msg.get("Date"))),
        "has_message_id":         int(bool(msg.get("Message-ID"))),
        "received_hops":          received_hops,
        "date_is_future":         date_is_future,
        "subject_all_caps":       subject_all_caps,
        "subject_caps_ratio":     subject_caps_ratio,
        "subject_money":          subject_money,
        "subject_exclamation":    subject_exclamation,
        "subject_has_numbers":    subject_has_numbers,
        "subject_has_special":    subject_has_special,
        "subject_length":         len(subject),
        "subject_word_count":     len(subject.split()),
        "dollar_count":           dollar_count,
        "total_money_symbols":    total_money,
        "has_script":             has_script,
        "has_iframe":             has_iframe,
        "has_form":               has_form,
        "url_count":              len(all_urls),
        "url_has_ip":             int(ip_in_url > 0),
        "url_suspicious_tlds":    suspicious_tld_count,
        "url_mismatch_count":     mismatches,
        "subject_text":           subject,
    }


# ══════════════════════════════════════════════════════════════════════════
# Per-file processing
# ══════════════════════════════════════════════════════════════════════════
def process_eml_file(path: Path, label: int) -> dict | None:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        msg = BytesParser(policy=policy.compat32).parsebytes(raw)
    except Exception as e:
        log.warning(f"Skipping {path.name}: parse error: {e}")
        return None

    try:
        body = extract_body(msg)
    except Exception as e:
        log.warning(f"Skipping {path.name}: body extraction error: {e}")
        return None

    try:
        feats = semantic_features(msg, body)
    except Exception as e:
        log.warning(f"Skipping {path.name}: feature extraction error: {e}")
        return None

    feats["body_text"] = body
    feats["label"] = label
    return feats


def process_dir(dirpath: Path, label: int) -> list[dict]:
    rows = []
    if not dirpath.exists():
        log.warning(f"Directory not found: {dirpath}")
        return rows

    files = sorted([p for p in dirpath.iterdir() if p.is_file()])
    log.info(f"Processing {len(files)} files in {dirpath} (label={label})")

    skipped = 0
    for p in files:
        row = process_eml_file(p, label)
        if row is None:
            skipped += 1
            continue
        # skip empty bodies — zero-content samples are not useful
        if not row["body_text"].strip():
            skipped += 1
            continue
        rows.append(row)

    log.info(f"  -> {len(rows)} extracted, {skipped} skipped")
    return rows


# ══════════════════════════════════════════════════════════════════════════
# CSV output / append
# ══════════════════════════════════════════════════════════════════════════
def write_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, 0) for c in COLUMNS})
    log.info(f"Wrote {len(rows)} rows -> {out_path}")


def append_to_existing(new_csv: Path, target_csv: Path):
    if not target_csv.exists():
        log.error(f"Target dataset not found: {target_csv}")
        sys.exit(1)

    # verify header matches
    with open(target_csv, newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f))
    if existing_header != COLUMNS:
        log.error("Header mismatch between target dataset and expected schema!")
        log.error(f"  target  : {existing_header}")
        log.error(f"  expected: {COLUMNS}")
        sys.exit(1)

    backup = target_csv.with_suffix(target_csv.suffix + ".bak")
    shutil.copy2(target_csv, backup)
    log.info(f"Backup created: {backup}")

    with open(new_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        new_header = next(reader)
        new_rows = list(reader)
    if new_header != COLUMNS:
        log.error("New samples CSV header mismatch — aborting append.")
        sys.exit(1)

    with open(target_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    log.info(f"Appended {len(new_rows)} rows -> {target_csv}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Extract .eml files into gw_final_dataset.csv schema")
    ap.add_argument("--phishing-dir", type=Path, help="Folder of phishing .eml files (label=1)")
    ap.add_argument("--legit-dir",    type=Path, help="Folder of legitimate .eml files (label=0)")
    ap.add_argument("--out",          type=Path, required=True, help="Output CSV path for new samples")
    ap.add_argument("--append",       type=Path, help="Existing gw_final_dataset.csv to append new rows to")
    args = ap.parse_args()

    if not args.phishing_dir and not args.legit_dir:
        log.error("Provide at least one of --phishing-dir / --legit-dir")
        sys.exit(1)

    all_rows = []
    if args.phishing_dir:
        all_rows.extend(process_dir(args.phishing_dir, label=1))
    if args.legit_dir:
        all_rows.extend(process_dir(args.legit_dir, label=0))

    if not all_rows:
        log.error("No rows extracted — nothing to write.")
        sys.exit(1)

    n_phish = sum(1 for r in all_rows if r["label"] == 1)
    n_legit = sum(1 for r in all_rows if r["label"] == 0)
    log.info(f"Total new samples: {len(all_rows)} (phishing={n_phish}, legit={n_legit})")

    write_csv(all_rows, args.out)

    if args.append:
        append_to_existing(args.out, args.append)
        log.info("Done. Re-run train_models.py to retrain on the combined dataset.")


if __name__ == "__main__":
    main()

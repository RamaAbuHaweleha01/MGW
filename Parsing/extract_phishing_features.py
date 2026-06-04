#!/usr/bin/env python3
"""
~/MGW/Parsing/extract_phishing_features.py
Comprehensive Email Feature Extractor

Two modes:
  1. Offline / dataset mode  — reads a CSV, extracts features, writes output CSV
  2. Live / pipeline mode    — called from mail_filter.py with an email.message object

Three extraction tracks:
  Track A — Semantic metadata (URLs, code, keywords, auth signals)
  Track B — Structural NLP-ready text (cleaned, tokenised)
  Track C — Attachment analysis — extracts DECODED BINARY BYTES for sandbox dispatch

FIX (critical): Track C previously stored payload_b64 (a base64-encoded string).
mail_filter was then passing that string — not the real bytes — to sandbox_client,
which caused CAPE to receive a tiny text file instead of the executable.
Track C now stores payload_bytes (raw bytes) directly. mail_filter reads
payload_bytes and passes them to sandbox_client.analyze_attachment().
"""
from __future__ import annotations
import os, sys, re, json, html, logging
from datetime import datetime
from pathlib import Path
from email.utils import parseaddr, parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd
import numpy  as np
import warnings
warnings.filterwarnings("ignore")

MGW_ROOT = Path.home() / "MGW"

logger = logging.getLogger("feature_extractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Extension maps (must stay in sync with sandbox_client.py) ────────────────
WIN_ONLY_EXTS = {".exe",".msi",".bat",".cmd",".ps1",".vbs",".lnk"}
LIN_ONLY_EXTS = {".sh",".elf",".run",".deb",".rpm",".bin"}
BOTH_EXTS     = {".js",".zip",".7z",".rar",".iso",".img",
                 ".docm",".xlsm",".pptm",".rtf",".pdf",".html",".htm",".link"}
ARCHIVE_EXTS  = {".zip",".7z",".rar",".iso",".img"}
OFFICE_EXTS   = {".docm",".xlsm",".pptm",".rtf",".pdf"}
SCRIPT_EXTS   = {".js",".vbs",".ps1",".bat",".cmd",".sh"}

BAD_TLDS    = {".tk",".ml",".ga",".cf",".gq",".xyz",".top",
               ".club",".online",".site",".work",".date",".loan"}
FREE_EMAILS = {"gmail.com","yahoo.com","hotmail.com","outlook.com","aol.com"}

PHISHING_KEYWORDS = [
    "urgent","verify","account","bank","paypal","suspended","click",
    "login","password","credit","social security","ssn","limited",
    "unusual","activity","confirm","update","security","fraud","claim",
    "prize","winner","lottery","inheritance","million","billion",
    "dollars","transfer","western union","money gram","wire transfer",
    "bank account","routing number","credit card","debit card",
    "expire","deadline","immediately","action required","restricted",
    "blocked","terminated","unauthorized","validate","credentials",
]


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A — Semantic feature extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_track_a(msg_or_row, body_text: str = "") -> dict:
    """
    Extract semantic / phishing-signal features from a live email message
    (email.message.Message) or a dataset row (pandas Series).
    Always returns a flat dict of numeric features.
    """
    features: dict = {}
    text  = body_text or ""
    lower = text.lower()

    # ── URL features ──────────────────────────────────────────────────────────
    url_re    = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s<>"]*)?', re.I)
    href_re   = re.compile(r'href=["\']([^"\']+)["\']', re.I)
    all_urls  = url_re.findall(text)
    href_urls = href_re.findall(text)

    mismatch_re = re.compile(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', re.I)
    mismatches = sum(
        1 for m in mismatch_re.finditer(text)
        if m.group(2).strip().startswith("http")
        and m.group(2).strip() not in m.group(1)
    )

    ip_in_url      = sum(1 for u in all_urls
                         if re.search(r'https?://\d+\.\d+\.\d+\.\d+', u))
    encoded_urls   = sum(1 for u in all_urls if '%' in u)
    susp_tld_count = sum(
        1 for u in all_urls
        if '/' in u and len(u.split('/')) > 2
        and any(u.lower().split('/')[2].endswith(t) for t in BAD_TLDS)
    )

    parsed_urls  = [urlparse(u) for u in all_urls]
    https_count  = sum(1 for p in parsed_urls if p.scheme == "https")
    http_count   = sum(1 for p in parsed_urls if p.scheme == "http")
    url_len_vals = [len(u) for u in all_urls] or [0]
    max_dots     = max((u.count('.') for u in all_urls), default=0)
    has_subdomain= sum(1 for p in parsed_urls if p.netloc.count('.') > 1)

    features.update({
        "url_count":               len(all_urls),
        "href_url_count":          len(href_urls),
        "url_mismatch_count":      mismatches,
        "url_has_ip":              int(ip_in_url > 0),
        "url_encoded_count":       encoded_urls,
        "url_suspicious_tlds":     susp_tld_count,
        "url_avg_length":          float(np.mean(url_len_vals)),
        "url_max_length":          max(url_len_vals),
        "url_count_https":         https_count,
        "url_count_http":          http_count,
        "url_has_https":           int(https_count > 0),
        "url_has_http":            int(http_count  > 0),
        "url_max_dots":            max_dots,
        "url_has_subdomains":      int(has_subdomain > 0),
        "url_has_at_symbol":       int(any('@' in u for u in all_urls)),
        "url_has_percent_encoding":int(encoded_urls > 0),
    })

    # ── Embedded code ─────────────────────────────────────────────────────────
    features.update({
        "has_script":         int(bool(re.search(r'<script',          text, re.I))),
        "has_onclick":        int(bool(re.search(r'onclick\s*=',       text, re.I))),
        "has_onload":         int(bool(re.search(r'onload\s*=',        text, re.I))),
        "has_iframe":         int(bool(re.search(r'<iframe',           text, re.I))),
        "has_base64":         int(bool(re.search(r'base64[,\s]',       text, re.I))),
        "has_data_uri":       int(bool(re.search(r'data:[^;]+;base64', text, re.I))),
        "has_eval":           int(bool(re.search(r'\beval\s*\(',       text, re.I))),
        "has_unescape":       int(bool(re.search(r'unescape\s*\(',     text, re.I))),
        "has_form":           int(bool(re.search(r'<form',             text, re.I))),
        "has_input_password": int(bool(re.search(
            r'<input[^>]+type=["\']?password', text, re.I))),
        "has_javascript":     int(bool(re.search(r'javascript:', text, re.I))),
        "has_html_tags":      int(bool(re.search(r'<[a-zA-Z][^>]*>', text))),
        "html_tag_count":     len(re.findall(r'<[a-zA-Z][^>]*>', text)),
    })

    # ── HTML obfuscation ──────────────────────────────────────────────────────
    entities = re.findall(r'&[#a-zA-Z0-9]+;', text)
    features.update({
        "html_entity_count": len(entities),
        "has_html_entities": int(len(entities) > 0),
        "obfuscated_chars":  int(html.unescape(text) != text),
    })

    # ── Phishing keywords ─────────────────────────────────────────────────────
    kw_counts = {f"keyword_{k.replace(' ','_')}": lower.count(k)
                 for k in PHISHING_KEYWORDS}
    total_kw  = sum(kw_counts.values())
    uniq_kw   = sum(1 for v in kw_counts.values() if v > 0)
    features.update(kw_counts)
    features.update({
        "total_phishing_keywords":  total_kw,
        "unique_phishing_keywords": uniq_kw,
    })

    # ── Composite psychological scores ────────────────────────────────────────
    features["urgency_score"] = min(1.0, sum(lower.count(w) for w in [
        "urgent","immediately","asap","deadline","expire",
        "limited time","action required","hours remaining"]) * 0.2)
    features["fear_score"]    = min(1.0, sum(lower.count(w) for w in [
        "suspended","terminated","closed","blocked","restricted",
        "unauthorized","fraud","compromised","hacked"]) * 0.2)
    features["curiosity_score"] = min(1.0, sum(lower.count(w) for w in [
        "winner","won","prize","selected","chosen","lucky","claim",
        "congratulations","inheritance","lottery"]) * 0.2)

    # ── Body statistics ───────────────────────────────────────────────────────
    words      = text.split()
    lines      = text.splitlines()
    paragraphs = [p for p in re.split(r'\n{2,}', text) if p.strip()]
    features.update({
        "body_length":          len(text),
        "body_word_count":      len(words),
        "body_line_count":      len(lines),
        "body_paragraph_count": len(paragraphs),
        "avg_word_length":      float(np.mean([len(w) for w in words]))
                                if words else 0.0,
        "unique_word_count":    len(set(w.lower() for w in words)),
        "unique_word_ratio":    len(set(w.lower() for w in words))
                                / max(len(words), 1),
        "caps_ratio":           sum(1 for c in text if c.isupper())
                                / max(len(text), 1),
        "exclamation_count":    text.count('!'),
        "question_count":       text.count('?'),
        "dollar_sign_count":    text.count('$'),
        "euro_sign_count":      text.count('€'),
        "pound_sign_count":     text.count('£'),
        "yen_sign_count":       text.count('¥'),
        "total_money_symbols":  text.count('$') + text.count('€') + text.count('£'),
    })

    # ── Email addresses in body ───────────────────────────────────────────────
    email_matches = re.findall(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', text)
    features.update({
        "email_in_body_count":        len(email_matches),
        "unique_email_in_body_count": len(set(email_matches)),
    })

    # ── Phone / IP counts ─────────────────────────────────────────────────────
    features["phone_count"]      = len(
        re.findall(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text))
    features["ip_address_count"] = len(
        re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text))

    return features


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A (header section) — from live email message object
# ══════════════════════════════════════════════════════════════════════════════
def extract_header_features(message) -> dict:
    """
    Extract features from email.message.Message header fields.
    Returns numeric feature dict + raw subject_text for NLP.
    """
    features = {}

    from_addr   = message.get("From",         "") or ""
    reply_to    = message.get("Reply-To",      "") or ""
    return_path = message.get("Return-Path",   "") or ""
    subject     = message.get("Subject",       "") or ""
    date_str    = message.get("Date",          "") or ""

    # From field
    from_clean  = parseaddr(from_addr)[1] or from_addr
    from_domain = from_clean.split('@')[-1].strip(">") if '@' in from_clean else ""
    features.update({
        "from_domain_length":    len(from_domain),
        "from_localpart_length": len(from_clean.split('@')[0])
                                 if '@' in from_clean else 0,
        "has_numeric_in_domain": int(any(c.isdigit() for c in from_domain)),
        "suspicious_tld_sender": int(any(from_domain.endswith(t)
                                         for t in BAD_TLDS)),
        "is_free_email":         int(from_domain in FREE_EMAILS),
    })

    # Reply-To / Return-Path mismatch
    reply_clean  = parseaddr(reply_to)[1] or reply_to
    reply_domain = reply_clean.split('@')[-1] if '@' in reply_clean else ""
    ret_clean    = return_path.strip("<>")
    ret_domain   = ret_clean.split('@')[-1] if '@' in ret_clean else ""
    features.update({
        "has_reply_to":                 int(bool(reply_to)),
        "reply_to_domain_matches_from": int(reply_domain == from_domain),
        "has_return_path":              int(bool(return_path)),
        "return_path_matches_from":     int(ret_domain == from_domain),
        "domain_mismatch":              int(
            (bool(reply_to)    and reply_domain != from_domain) or
            (bool(return_path) and ret_domain   != from_domain)
        ),
    })

    # Auth
    auth     = (message.get("Authentication-Results", "") or "").lower()
    has_dkim = int(bool(message.get("DKIM-Signature")))
    features.update({
        "has_auth_results": int(bool(auth)),
        "spf_fail":         int("spf=fail"   in auth),
        "dkim_fail":        int("dkim=fail"  in auth),
        "dmarc_fail":       int("dmarc=fail" in auth),
        "has_dkim":         has_dkim,
    })

    # Subject — numeric features + raw text for NLP
    subj_low = subject.lower()
    features.update({
        "subject_text":          subject,    # raw text → passed to NLP models
        "subject_length":        len(subject),
        "subject_word_count":    len(subject.split()),
        "subject_has_urgent":    int(bool(re.search(
            r'urgent|immediate|asap', subj_low))),
        "subject_has_verify":    int(bool(re.search(
            r'verify|confirm|validate', subj_low))),
        "subject_has_alert":     int("alert" in subj_low),
        "subject_has_suspended": int(bool(re.search(
            r'suspended|locked|closed', subj_low))),
        "subject_has_account":   int(bool(re.search(
            r'account|password|credential', subj_low))),
        "subject_all_caps":      int(subject.isupper() and len(subject) > 3),
        "subject_caps_ratio":    sum(1 for c in subject if c.isupper())
                                 / max(len(subject), 1),
        "subject_money":         int(any(s in subject
                                         for s in ["$","€","£","¥","money","wire"])),
        "subject_exclamation":   subject.count("!"),
        "subject_has_numbers":   int(any(c.isdigit() for c in subject)),
        "subject_has_special":   int(any(c in "!@#$%^&*()" for c in subject)),
        "subject_has_reply":     int(subj_low.startswith("re:")),
        "subject_has_fwd":       int(subj_low.startswith("fwd:")),
    })

    # Date
    features["has_date"]       = int(bool(date_str))
    features["date_is_future"] = 0
    features["date_is_weekend"]= 0
    features["hour_sent"]      = -1
    if date_str:
        try:
            date_obj = parsedate_to_datetime(date_str)
            features["date_is_future"]  = int(
                date_obj > datetime.now(date_obj.tzinfo))
            features["date_is_weekend"] = int(date_obj.weekday() >= 5)
            features["hour_sent"]       = date_obj.hour
        except Exception:
            pass

    # Structure flags
    features.update({
        "has_from":       int(bool(message.get("From"))),
        "has_to":         int(bool(message.get("To"))),
        "has_cc":         int(bool(message.get("Cc"))),
        "has_bcc":        int(bool(message.get("Bcc"))),
        "has_subject":    int(bool(subject)),
        "has_message_id": int(bool(message.get("Message-ID"))),
        "received_hops":  len(message.get_all("Received", [])),
        "received_count_normalized": min(
            len(message.get_all("Received", [])) / 10.0, 1.0),
    })

    return features


# ══════════════════════════════════════════════════════════════════════════════
# TRACK B — Structural text cleaning
# ══════════════════════════════════════════════════════════════════════════════
def extract_track_b(raw_body: str) -> str:
    """
    Clean raw body HTML into NLP-ready text.
    Preserve semantic placeholders (URL_TOKEN, EMAIL_TOKEN, etc.).
    """
    text = raw_body
    text = re.sub(r'https?://\S+',                  ' URL_TOKEN ',   text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', ' EMAIL_TOKEN ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  ' IP_TOKEN ',    text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\$€£¥]\s*[\d,]+',             ' MONEY_TOKEN ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TRACK C — Attachment extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_track_c(message) -> dict:
    """
    Walk email MIME parts and extract every attachment.

    CRITICAL FIX:
    Previously this function stored base64-encoded strings in 'payload_b64'.
    mail_filter was decoding that string and passing it to sandbox_client —
    but the decoding was broken, so CAPE received a 13-byte text file.

    Now: 'payload_bytes' holds the RAW DECODED BINARY (bytes object) returned
    directly by part.get_payload(decode=True). mail_filter passes payload_bytes
    straight to sandbox_client.analyze_attachment() — no extra encoding step.

    The 'sandbox_pipeline' list each entry contains:
      {
        filename     : str   — original filename for CAPE submission
        payload_bytes: bytes — real decoded binary content
        targets      : list  — ["windows"] / ["linux"] / ["windows","linux"]
        content_type : str
        size_bytes   : int
      }
    """
    attachments      = []
    sandbox_pipeline = []
    has_exec = has_archive = has_office = has_script = has_pdf = has_img = 0
    att_count = 0

    for part in message.walk():
        disp     = part.get_content_disposition() or ""
        ct       = part.get_content_type()
        filename = part.get_filename() or ""

        if "attachment" not in disp and not filename:
            continue

        att_count += 1
        ext = os.path.splitext(filename.lower())[1] if filename else ""

        if ext in WIN_ONLY_EXTS | LIN_ONLY_EXTS: has_exec    = 1
        if ext in ARCHIVE_EXTS:                   has_archive = 1
        if ext in OFFICE_EXTS:                    has_office  = 1
        if ext in SCRIPT_EXTS:                    has_script  = 1
        if ext == ".pdf":                         has_pdf     = 1
        if ct.startswith("image/"):               has_img     = 1

        # Platform routing from extension
        if   ext in WIN_ONLY_EXTS: targets = ["windows"]
        elif ext in LIN_ONLY_EXTS: targets = ["linux"]
        elif ext in BOTH_EXTS:     targets = ["windows", "linux"]
        else:                       targets = ["windows", "linux"]

        # FIX: get_payload(decode=True) returns the actual decoded binary bytes.
        # This is what CAPE needs — not a base64 string, not a filename path.
        payload_bytes = part.get_payload(decode=True)

        if payload_bytes:
            safe_name = filename or f"attachment_{att_count}"
            sandbox_pipeline.append({
                "filename":     safe_name,
                "payload_bytes": payload_bytes,   # ← real binary bytes
                "targets":      targets,
                "content_type": ct,
                "size_bytes":   len(payload_bytes),
            })

        attachments.append({
            "filename":     filename,
            "extension":    ext,
            "content_type": ct,
            "size_bytes":   len(payload_bytes) if payload_bytes else 0,
            "targets":      targets,
        })

    return {
        "attachment_count":            att_count,
        "has_executable_attachment":   has_exec,
        "has_archive_attachment":      has_archive,
        "has_office_macro_attachment": has_office,
        "has_script_attachment":       has_script,
        "has_pdf_attachment":          has_pdf,
        "has_image_attachment":        has_img,
        "sandbox_pipeline":            sandbox_pipeline,  # consumed by mail_filter
        "attachments_meta":            attachments,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Live pipeline entry point — called from mail_filter.py
# ══════════════════════════════════════════════════════════════════════════════
def extract_features_from_email(message) -> dict:
    """
    Full real-time feature extraction for a live email.message.Message object.

    Returns
    -------
    {
      "semantic_meta": dict   — merged Track A + header features → header.py + body.py
      "clean_text":    str    — Track B cleaned text              → body.py NLP
      "track_c":       dict   — attachment features + sandbox_pipeline list
                                sandbox_pipeline entries contain payload_bytes (bytes)
                                ready to pass directly to sandbox_client.analyze_attachment()
    }
    """
    raw_body = ""
    try:
        if message.is_multipart():
            for part in message.walk():
                ct = part.get_content_type()
                if ct in ("text/plain", "text/html"):
                    payload = part.get_payload(decode=True)
                    if payload:
                        raw_body += payload.decode("utf-8", errors="replace") + "\n"
        else:
            payload = message.get_payload(decode=True)
            if payload:
                raw_body = payload.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning(f"Body extraction error: {exc}")

    header_feats  = extract_header_features(message)
    body_a_feats  = extract_track_a(message, raw_body)
    clean_text    = extract_track_b(raw_body)
    track_c       = extract_track_c(message)

    # Merge all semantic signals into one dict for both header.py and body.py
    semantic_meta = {**body_a_feats, **header_feats}

    return {
        "semantic_meta": semantic_meta,
        "clean_text":    clean_text,
        "track_c":       track_c,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Offline / dataset mode
# ══════════════════════════════════════════════════════════════════════════════
class PhishingEmailFeatureExtractor:
    """Extract comprehensive features from a CSV email dataset (offline mode)."""

    def __init__(self, csv_file):
        self.csv_file    = csv_file
        self.df          = None
        self.features_df = None

    def load_dataset(self):
        # FIX: removed engine='python' + low_memory=False conflict.
        # Try multiple encodings; use default C engine (fast, handles large files).
        for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                self.df = pd.read_csv(self.csv_file, encoding=enc,
                                      on_bad_lines="skip",
                                      encoding_errors="replace")
                logger.info(
                    f"Loaded {self.csv_file} ({enc}) — {len(self.df)} rows")
                return self.df
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                logger.warning(f"Failed to load {self.csv_file}: {exc}")
                break
        raise RuntimeError(f"Cannot load {self.csv_file} with any encoding")

    def _extract_row(self, row) -> dict:
        text = ""
        for col in ["body", "text", "message", "email_text", "Body", "Text"]:
            if col in row.index and pd.notna(row.get(col)):
                text = str(row[col]); break

        feats = extract_track_a(None, text)

        subject = ""
        for sc in ["subject", "Subject"]:
            v = row.get(sc, None)
            if v is not None and pd.notna(v):
                subject = str(v); break

        from_v = ""
        for fc in ["from", "From"]:
            v = row.get(fc, None)
            if v is not None and pd.notna(v):
                from_v = str(v); break

        feats["subject_text"]       = subject
        feats["subject_length"]     = len(subject)
        feats["subject_word_count"] = len(subject.split())
        feats["from_length"]        = len(from_v)
        feats["has_from"]           = int(bool(from_v))
        feats["has_subject"]        = int(bool(subject))
        feats["clean_text"]         = extract_track_b(text)

        # Attachment columns (dataset-sourced — no binary payload available)
        feats["attachment_count"]          = 0
        feats["has_executable_attachment"] = 0
        feats["has_archive_attachment"]    = 0
        feats["has_pdf_attachment"]        = 0
        feats["has_document_attachment"]   = 0
        feats["has_image_attachment"]      = 0
        for att_col in ["attachment", "attachments", "has_attachment"]:
            if att_col in row.index and pd.notna(row.get(att_col)):
                val = str(row[att_col]).lower()
                feats["attachment_count"] = int(
                    bool(val and val not in ("0", "false", "none", "")))
                break

        return feats

    def extract_all(self) -> pd.DataFrame:
        if self.df is None:
            self.load_dataset()
        rows = []
        for _, row in self.df.iterrows():
            try:
                rows.append(self._extract_row(row))
            except Exception as exc:
                logger.warning(f"Row extraction error: {exc}")
                rows.append({})

        self.features_df = pd.DataFrame(rows).fillna(0)

        # Carry label column through
        for lc in ["label", "Label", "spam", "class"]:
            if lc in self.df.columns:
                self.features_df["label"] = self.df[lc].values
                break

        return self.features_df

    def run_pipeline(self, output_prefix: str = "phishing_features") -> pd.DataFrame:
        self.load_dataset()
        feats = self.extract_all()

        # Save numeric-only CSV; keep text columns in a separate file for NLP
        text_cols = [c for c in feats.columns if feats[c].dtype == object]
        feats_num = feats.drop(columns=text_cols, errors="ignore")
        feats_num.to_csv(f"{output_prefix}_features.csv", index=False)
        logger.info(
            f"Saved {output_prefix}_features.csv  "
            f"({len(feats_num)} rows, {len(feats_num.columns)} features)")

        # Also save text columns for NLP training
        if text_cols:
            feats[text_cols + (["label"] if "label" in feats.columns else [])].to_csv(
                f"{output_prefix}_text.csv", index=False)
            logger.info(f"Saved {output_prefix}_text.csv  ({len(feats)} rows)")

        return feats


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract features from phishing email dataset")
    ap.add_argument("csv_file", help="Path to input CSV")
    ap.add_argument("--output", "-o", default="phishing_features",
                    help="Output file prefix (default: phishing_features)")
    args = ap.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    ext = PhishingEmailFeatureExtractor(args.csv_file)
    ext.run_pipeline(output_prefix=args.output)
    print(f"Output: {args.output}_features.csv  /  {args.output}_text.csv")


if __name__ == "__main__":
    main()

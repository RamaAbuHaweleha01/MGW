#!/usr/bin/env python3
"""
~/MGW/mail_filter.py
Mail Gateway Filter — Main Controller (v2 — CAPEv2 integrated)

Pipeline:
  Track A (Semantic)    — phishing signal extraction
  Track B (Structural)  — text cleaning for NLP/ML
  Track C (Attachment)  — CAPEv2 sandbox detonation via local API

Scoring:
  All model results are REPORTED and LOGGED.
  No combined threshold is applied — every email is FORWARDED regardless.
  The final decision stage is deferred (set ENFORCE_THRESHOLD=1 to re-enable).

Attachment logs saved to:  ~/MGW/Attachment/<timestamp>_<filename>.json
Main filter log:           ~/MGW/mail_filter.log
Sandbox dispatch log:      ~/MGW/sandbox.log
"""
from __future__ import annotations
import sys, os, json, logging, asyncio, time, re, html, importlib.util
from datetime import datetime
from email import message_from_bytes
import smtplib

# ─── Paths ────────────────────────────────────────────────────────────────────
MGW_ROOT       = os.path.expanduser("~/MGW")
HEADER_SCRIPT  = os.path.join(MGW_ROOT, "models", "Header",  "header.py")
BODY_SCRIPT    = os.path.join(MGW_ROOT, "models", "Body",    "body.py")
SANDBOX_SCRIPT = os.path.join(MGW_ROOT, "models", "Attachment", "sandbox_client.py")
LOG_FILE       = os.path.join(MGW_ROOT, "mail_filter.log")
SANDBOX_LOG    = os.path.join(MGW_ROOT, "Attachment", "sandbox.log")
ATTACHMENT_DIR = os.path.join(MGW_ROOT, "Attachment")

os.makedirs(ATTACHMENT_DIR, exist_ok=True)

# ─── SMTP settings ────────────────────────────────────────────────────────────
MGW_LISTEN_HOST  = "0.0.0.0"
MGW_LISTEN_PORT  = 10025
MAIL_SERVER_HOST = "127.0.0.1"
MAIL_SERVER_PORT = 10026

# ─── Scoring control ─────────────────────────────────────────────────────────
# Set ENFORCE_THRESHOLD=1 in environment to re-enable DROP decisions.
# Default: report only, always forward.
ENFORCE_THRESHOLD = os.environ.get("ENFORCE_THRESHOLD", "0") == "1"
RISK_THRESHOLD    = 0.70

# ─── Resource thresholds ─────────────────────────────────────────────────────
CPU_THRESHOLD_PCT = 75.0
RAM_THRESHOLD_PCT = 80.0

# ─── Extension sets ──────────────────────────────────────────────────────────
WIN_ONLY_EXTS = {".exe",".msi",".bat",".cmd",".ps1",".vbs",".lnk"}
LIN_ONLY_EXTS = {".sh",".elf",".run",".deb",".rpm",".bin"}
BOTH_EXTS     = {
    ".js",".zip",".7z",".rar",".iso",".img",
    ".docm",".xlsm",".pptm",".rtf",".pdf",
    ".html",".htm",".link",
}
ARCHIVE_EXTS  = {".zip",".7z",".rar",".iso",".img"}
OFFICE_EXTS   = {".docm",".xlsm",".pptm",".rtf",".pdf"}
SCRIPT_EXTS   = {".js",".vbs",".ps1",".bat",".cmd",".sh"}

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mail_filter")

s_handler = logging.FileHandler(SANDBOX_LOG)
s_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
sandbox_logger = logging.getLogger("sandbox_dispatch")
sandbox_logger.addHandler(s_handler)
sandbox_logger.setLevel(logging.INFO)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def _import_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resource_health() -> dict:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.3)
        ram = psutil.virtual_memory().percent
        return {"cpu_pct": cpu, "ram_pct": ram, "healthy": True}
    except ImportError:
        return {"cpu_pct": 0.0, "ram_pct": 0.0, "healthy": False}


def _use_roberta(health: dict) -> bool:
    if not health["healthy"]:
        return True
    overloaded = (
        health["cpu_pct"] > CPU_THRESHOLD_PCT or
        health["ram_pct"] > RAM_THRESHOLD_PCT
    )
    return not overloaded


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A — Semantic Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def semantic_track(msg, raw_body: str) -> dict:
    subject  = msg.get("Subject", "") or ""
    text     = raw_body
    lower    = text.lower()
    subj_low = subject.lower()

    url_pattern  = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s<>"]*)?', re.I)
    href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.I)
    all_urls     = url_pattern.findall(text)
    href_urls    = href_pattern.findall(text)

    mismatch_pattern = re.compile(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', re.I)
    mismatches = sum(
        1 for m in mismatch_pattern.finditer(text)
        if m.group(2).strip().startswith("http")
        and m.group(2).strip() not in m.group(1)
    )

    ip_in_url    = sum(1 for u in all_urls
                       if re.search(r'https?://\d+\.\d+\.\d+\.\d+', u))
    encoded_urls = sum(1 for u in all_urls if '%' in u)
    bad_tlds     = {'.tk','.ml','.ga','.cf','.gq','.xyz','.top',
                    '.club','.online','.site','.work','.date','.loan'}
    suspicious_tld_count = sum(
        1 for u in all_urls
        if '/' in u and len(u.split('/')) > 2
        and any(u.lower().split('/')[2].endswith(t) for t in bad_tlds))

    has_script     = int(bool(re.search(r'<script',          text, re.I)))
    has_onclick    = int(bool(re.search(r'onclick\s*=',       text, re.I)))
    has_onload     = int(bool(re.search(r'onload\s*=',        text, re.I)))
    has_iframe     = int(bool(re.search(r'<iframe',           text, re.I)))
    has_base64     = int(bool(re.search(r'base64[,\s]',       text, re.I)))
    has_data_uri   = int(bool(re.search(r'data:[^;]+;base64', text, re.I)))
    has_eval       = int(bool(re.search(r'\beval\s*\(',       text, re.I)))
    has_unescape   = int(bool(re.search(r'unescape\s*\(',     text, re.I)))
    has_form       = int(bool(re.search(r'<form',             text, re.I)))
    has_input_pass = int(bool(re.search(
        r'<input[^>]+type=["\']?password', text, re.I)))

    html_entity_count = len(re.findall(r'&[#a-zA-Z0-9]+;', text))
    decoded           = html.unescape(text)
    obfuscated_chars  = int(decoded != text)

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
    keyword_counts     = {f"kw_{k.replace(' ','_')}": lower.count(k)
                          for k in PHISHING_KEYWORDS}
    total_phishing_kw  = sum(keyword_counts.values())
    unique_phishing_kw = sum(1 for v in keyword_counts.values() if v > 0)

    urgency_score = min(1.0, sum(lower.count(w) for w in [
        "urgent","immediately","asap","deadline","expire",
        "limited time","action required","hours remaining"]) * 0.2)
    fear_score    = min(1.0, sum(lower.count(w) for w in [
        "suspended","terminated","closed","blocked","restricted",
        "unauthorized","fraud","compromised","hacked"]) * 0.2)
    curiosity_score = min(1.0, sum(lower.count(w) for w in [
        "winner","won","prize","selected","chosen","lucky","claim",
        "congratulations","inheritance","lottery"]) * 0.2)

    subject_has_urgent  = int(bool(re.search(r'urgent|immediate|asap', subj_low)))
    subject_has_verify  = int(bool(re.search(r'verify|confirm|validate', subj_low)))
    subject_has_alert   = int("alert" in subj_low)
    subject_all_caps    = int(subject.isupper() and len(subject) > 3)
    subject_caps_ratio  = (sum(1 for c in subject if c.isupper())
                           / max(len(subject), 1))
    subject_money       = int(any(s in subject for s in
                                  ["$","€","£","¥","money","wire"]))
    subject_exclamation = subject.count("!")
    subject_has_numbers = int(any(c.isdigit() for c in subject))
    subject_has_special = int(any(c in "!@#$%^&*()" for c in subject))

    auth_results = msg.get("Authentication-Results","") or ""
    auth_lower   = auth_results.lower()
    spf_fail     = int("spf=fail"  in auth_lower)
    dkim_fail    = int("dkim=fail" in auth_lower)
    dmarc_fail   = int("dmarc=fail" in auth_lower)
    has_dkim     = int(bool(msg.get("DKIM-Signature")))

    from_addr    = msg.get("From","") or ""
    reply_to     = msg.get("Reply-To","") or ""
    return_path  = msg.get("Return-Path","") or ""
    from_domain  = (from_addr.split("@")[-1].strip(">").strip()
                    if "@" in from_addr else "")
    reply_domain = (reply_to.split("@")[-1].strip(">").strip()
                    if "@" in reply_to else "")
    return_domain= (return_path.strip("<>").split("@")[-1].strip()
                    if "@" in return_path else "")
    domain_mismatch = int(
        (bool(reply_to) and reply_domain != from_domain) or
        (bool(return_path) and return_domain != from_domain)
    )
    suspicious_tld_sender  = int(any(from_domain.endswith(t) for t in bad_tlds))
    has_numeric_in_domain  = int(any(c.isdigit() for c in from_domain))
    received_hops          = len(msg.get_all("Received",[]))
    date_str               = msg.get("Date","") or ""
    date_is_future         = 0
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            date_obj = parsedate_to_datetime(date_str)
            date_is_future = int(date_obj > datetime.now(date_obj.tzinfo))
        except Exception:
            pass

    dollar_count = text.count("$")
    total_money  = dollar_count + text.count("€") + text.count("£")

    return {
        "url_count":             len(all_urls),
        "href_url_count":        len(href_urls),
        "url_mismatch_count":    mismatches,
        "url_has_ip":            int(ip_in_url > 0),
        "url_encoded_count":     encoded_urls,
        "url_suspicious_tlds":   suspicious_tld_count,
        "has_script":            has_script,
        "has_onclick":           has_onclick,
        "has_onload":            has_onload,
        "has_iframe":            has_iframe,
        "has_base64":            has_base64,
        "has_data_uri":          has_data_uri,
        "has_eval":              has_eval,
        "has_unescape":          has_unescape,
        "has_form":              has_form,
        "has_input_password":    has_input_pass,
        "html_entity_count":     html_entity_count,
        "obfuscated_chars":      obfuscated_chars,
        **keyword_counts,
        "total_phishing_keywords":  total_phishing_kw,
        "unique_phishing_keywords": unique_phishing_kw,
        "urgency_score":         urgency_score,
        "fear_score":            fear_score,
        "curiosity_score":       curiosity_score,
        "subject_has_urgent":    subject_has_urgent,
        "subject_has_verify":    subject_has_verify,
        "subject_has_alert":     subject_has_alert,
        "subject_all_caps":      subject_all_caps,
        "subject_caps_ratio":    subject_caps_ratio,
        "subject_money":         subject_money,
        "subject_exclamation":   subject_exclamation,
        "subject_has_numbers":   subject_has_numbers,
        "subject_has_special":   subject_has_special,
        "subject_length":        len(subject),
        "subject_word_count":    len(subject.split()),
        "subject_text":          subject,
        "spf_fail":              spf_fail,
        "dkim_fail":             dkim_fail,
        "dmarc_fail":            dmarc_fail,
        "has_dkim":              has_dkim,
        "domain_mismatch":       domain_mismatch,
        "suspicious_tld_sender": suspicious_tld_sender,
        "has_numeric_in_domain": has_numeric_in_domain,
        "received_hops":         received_hops,
        "date_is_future":        date_is_future,
        "dollar_count":          dollar_count,
        "total_money_symbols":   total_money,
        "has_from":              int(bool(msg.get("From"))),
        "has_to":                int(bool(msg.get("To"))),
        "has_cc":                int(bool(msg.get("Cc"))),
        "has_bcc":               int(bool(msg.get("Bcc"))),
        "has_subject":           int(bool(msg.get("Subject"))),
        "has_date":              int(bool(msg.get("Date"))),
        "has_message_id":        int(bool(msg.get("Message-ID"))),
        "has_reply_to":          int(bool(reply_to)),
        "has_return_path":       int(bool(return_path)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRACK B — Structural
# ══════════════════════════════════════════════════════════════════════════════
def structural_track(raw_body: str) -> str:
    text = raw_body
    text = re.sub(r'https?://\S+',                   ' URL_TOKEN ',   text)
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b',  ' EMAIL_TOKEN ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',   ' IP_TOKEN ',    text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\$€£¥]\s*[\d,]+',              ' MONEY_TOKEN ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TRACK C — Attachment → CAPEv2
# ══════════════════════════════════════════════════════════════════════════════
def attachment_track(msg) -> dict:
    """
    Extract all attachments and dispatch each to CAPEv2 via sandbox_client.
    Returns aggregate result dict for the verdict log.
    """
    import base64

    # Lazy-import sandbox_client so mail_filter works even if sandbox is down
    try:
        sc = _import_script(SANDBOX_SCRIPT, "sandbox_client")
        cape_available = True
    except Exception as exc:
        logger.error(f"sandbox_client import failed: {exc}")
        cape_available = False

    attachment_results = []

    for part in msg.walk():
        disp     = part.get_content_disposition() or ""
        ct       = part.get_content_type()
        filename = part.get_filename() or ""

        if "attachment" in disp or (
            filename and
            ct not in ("text/plain", "text/html", "multipart/mixed")
        ):
            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            ext = os.path.splitext(filename.lower())[1] if filename else ""
            logger.info(f"Attachment found: {filename} ({ct}) size={len(payload)}B")
            sandbox_logger.info(f"Dispatching to CAPE: {filename} ext={ext}")

            if cape_available:
                try:
                    result = sc.analyze_attachment(filename, payload)
                    attachment_results.append(result)
                    sandbox_logger.info(
                        f"CAPE result: {filename} verdict={result.get('verdict')} "
                        f"risk={result.get('risk_score',0):.4f} "
                        f"log={result.get('log_path','?')}"
                    )
                except Exception as exc:
                    logger.error(f"CAPE analysis error for {filename}: {exc}")
                    attachment_results.append({
                        "filename":   filename,
                        "verdict":    "error",
                        "risk_score": 0.50,
                        "error":      str(exc),
                    })
            else:
                attachment_results.append({
                    "filename":   filename,
                    "verdict":    "cape_unavailable",
                    "risk_score": 0.40,
                })

    if not attachment_results:
        return {"has_attachments": 0, "risk_probability": 0.0, "attachments": []}

    max_risk = max(a.get("risk_score", 0) for a in attachment_results)
    return {
        "has_attachments":  1,
        "attachment_count": len(attachment_results),
        "risk_probability": round(float(max_risk), 6),
        "attachments":      attachment_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Body extractor
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Model callers
# ══════════════════════════════════════════════════════════════════════════════
def analyze_header(semantic_meta: dict) -> dict:
    try:
        mod = _import_script(HEADER_SCRIPT, "header_analyzer")
        return mod.analyze(semantic_meta)
    except Exception as exc:
        logger.error(f"Header analysis failed: {exc}")
        return {"risk_probability": 0.5, "risk_factors": [str(exc)], "engine": "error"}


def analyze_body(clean_text: str, semantic_meta: dict, use_roberta: bool = True) -> dict:
    try:
        mod = _import_script(BODY_SCRIPT, "body_analyzer")
        return mod.analyze(clean_text, semantic_meta, use_roberta=use_roberta)
    except Exception as exc:
        logger.error(f"Body analysis failed: {exc}")
        return {"risk_probability": 0.5, "risk_factors": [str(exc)], "engine": "error"}


# ══════════════════════════════════════════════════════════════════════════════
# Verdict logger  (report only — no scoring weight applied)
# ══════════════════════════════════════════════════════════════════════════════
def log_verdict(meta: dict, header_r: dict, body_r: dict,
                attach_r: dict, action: str):
    ts       = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    h_prob   = header_r.get("risk_probability", 0.0)
    b_prob   = body_r.get("risk_probability",   0.0)
    a_prob   = attach_r.get("risk_probability", 0.0)
    h_engine = header_r.get("engine", "?")
    b_engine = body_r.get("engine",  "?")
    a_count  = attach_r.get("attachment_count", 0)

    # Per-attachment detail
    attach_lines = ""
    for a in attach_r.get("attachments", []):
        attach_lines += (
            f"    ├─ {a.get('filename','?'):30s} "
            f"verdict={a.get('verdict','?'):12s} "
            f"risk={a.get('risk_score',0):.4f}\n"
        )
        for v in a.get("cape_verdicts", []):
            attach_lines += (
                f"    │  [{v.get('platform','?'):8s}] "
                f"task={v.get('task_id','?')} "
                f"machine={v.get('machine','?')} "
                f"risk={v.get('risk_score',0):.4f} "
                f"behaviors={v.get('behaviors',[])}\n"
            )

    block = (
        f"\n{'='*65}\n"
        f"[{ts}] Message-ID: {meta['message_id']}\n"
        f"  From : {meta['from']}\n"
        f"  To   : {meta['to']}\n"
        f"  ── Model Results (report only — no weighting) ─────────\n"
        f"  Header  [{h_engine:20s}] risk = {h_prob:.6f}\n"
        f"  Body    [{b_engine:20s}] risk = {b_prob:.6f}\n"
        f"  Attach  [cape-sandbox        ] risk = {a_prob:.6f}  "
        f"(count={a_count})\n"
        f"{attach_lines}"
        f"  ── Threshold scoring disabled — all emails forwarded ───\n"
        f"  ACTION : {action}\n"
        f"{'='*65}"
    )
    logger.info(block)


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ══════════════════════════════════════════════════════════════════════════════
def process_message(raw_bytes: bytes):
    msg  = message_from_bytes(raw_bytes)
    meta = {
        "message_id": msg.get("Message-ID", f"<gen-{time.time()}@mgw>"),
        "from":       msg.get("From",  "unknown"),
        "to":         msg.get("To",    "unknown"),
    }

    health      = _resource_health()
    use_roberta = _use_roberta(health)
    logger.info(
        f"Resource — CPU={health['cpu_pct']:.1f}%  "
        f"RAM={health['ram_pct']:.1f}%  "
        f"body_engine={'roberta' if use_roberta else 'tfidf'}"
    )

    raw_body      = extract_body(msg)
    semantic_meta = semantic_track(msg, raw_body)
    clean_text    = structural_track(raw_body)

    # Track C — CAPEv2 sandbox (runs first so report is available for logging)
    attach_result = attachment_track(msg)
    if attach_result["has_attachments"]:
        logger.info(
            f"Attachments: {attach_result['attachment_count']}  "
            f"max_risk={attach_result['risk_probability']:.4f}"
        )

    # Track A / B — models
    header_result = analyze_header(semantic_meta)
    body_result   = analyze_body(clean_text, semantic_meta, use_roberta=use_roberta)

    h_prob = header_result.get("risk_probability", 0.0)
    b_prob = body_result.get("risk_probability",   0.0)
    a_prob = attach_result.get("risk_probability", 0.0)

    # ── Decision ──────────────────────────────────────────────────────────────
    if ENFORCE_THRESHOLD:
        # Weighted score (weights TBD — placeholder equal weighting)
        combined = (h_prob + b_prob + a_prob) / 3.0
        if combined >= RISK_THRESHOLD:
            action = f"DROPPED (enforce_threshold=ON risk={combined:.4f})"
            logger.warning(f"[DROPPED] {meta['message_id']} risk={combined:.4f}")
        else:
            action = f"FORWARDED (enforce_threshold=ON risk={combined:.4f})"
            _forward(raw_bytes, meta["from"], meta["to"])
    else:
        # Report-only mode — always forward
        action = (
            f"FORWARDED (report-only mode  "
            f"header={h_prob:.4f}  body={b_prob:.4f}  attach={a_prob:.4f})"
        )
        _forward(raw_bytes, meta["from"], meta["to"])

    log_verdict(meta, header_result, body_result, attach_result, action)


def _forward(raw_bytes, from_addr, to_addr):
    try:
        with smtplib.SMTP(MAIL_SERVER_HOST, MAIL_SERVER_PORT, timeout=30) as s:
            s.sendmail(from_addr, [to_addr], raw_bytes)
        logger.info(f"Forwarded to {to_addr}")
    except Exception as exc:
        logger.error(f"Forwarding failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# SMTP proxy
# ══════════════════════════════════════════════════════════════════════════════
class SMTPHandler(asyncio.Protocol):
    def __init__(self):
        self._buf   = b""
        self._state = "INIT"
        self._from  = ""
        self._to    = []
        self._data  = b""
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        self._send("220 mgw.company.com ESMTP MailFilter ready")

    def data_received(self, data):
        self._buf += data
        while b"\r\n" in self._buf:
            line, self._buf = self._buf.split(b"\r\n", 1)
            self._handle(line.decode("utf-8", errors="replace"))

    def _send(self, text):
        self.transport.write((text + "\r\n").encode())

    def _handle(self, line):
        upper = line.upper()
        if self._state == "DATA_BODY":
            if line == ".":
                self._state = "DONE"
                self._send("250 OK: queued")
                try:
                    process_message(self._data)
                except Exception as exc:
                    logger.error(f"Pipeline error: {exc}")
            else:
                self._data += (
                    (line[1:] if line.startswith(".") else line) + "\r\n"
                ).encode()
            return
        if upper.startswith("EHLO") or upper.startswith("HELO"):
            self._send(f"250-mgw.company.com\r\n250 OK")
        elif upper.startswith("MAIL FROM"):
            self._from = line.split(":", 1)[1].strip().strip("<>")
            self._send("250 OK")
        elif upper.startswith("RCPT TO"):
            self._to.append(line.split(":", 1)[1].strip().strip("<>"))
            self._send("250 OK")
        elif upper == "DATA":
            self._state = "DATA_BODY"
            self._send("354 End data with <CR><LF>.<CR><LF>")
        elif upper == "QUIT":
            self._send("221 Bye")
            self.transport.close()
        elif upper == "RSET":
            self._from = ""; self._to = []; self._data = b""
            self._state = "INIT"; self._send("250 OK")
        else:
            self._send("500 Unrecognised command")

    def connection_lost(self, exc):
        if exc:
            logger.debug(f"Connection error: {exc}")


async def run_server():
    loop   = asyncio.get_event_loop()
    server = await loop.create_server(SMTPHandler, MGW_LISTEN_HOST, MGW_LISTEN_PORT)
    logger.info(
        f"MailFilter SMTP proxy listening on {MGW_LISTEN_HOST}:{MGW_LISTEN_PORT}"
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stdin":
        raw = sys.stdin.buffer.read()
        if raw:
            process_message(raw)
    else:
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            logger.info("MailFilter shutting down.")

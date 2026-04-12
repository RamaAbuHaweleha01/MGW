#!/usr/bin/env python3
"""
~/MGW/mail_filter.py
Mail Gateway Filter — Main Controller
Two-track preprocessing pipeline:
  Track A (Semantic)   — counts and preserves all phishing signals as metadata
  Track B (Structural) — cleans and tokenizes text for ML/NLP models
"""
from __future__ import annotations
import sys, os, json, logging, asyncio, time, re, html
from datetime import datetime
from email import message_from_bytes
from email.policy import default as email_policy
import smtplib, importlib.util

# ─── Paths ────────────────────────────────────────────────────────────────────
MGW_ROOT       = os.path.expanduser("~/MGW")
HEADER_SCRIPT  = os.path.join(MGW_ROOT, "models", "Header", "header.py")
BODY_SCRIPT    = os.path.join(MGW_ROOT, "models", "Body",   "body.py")
LOG_FILE       = os.path.join(MGW_ROOT, "mail_filter.log")

# ─── SMTP settings ────────────────────────────────────────────────────────────
MGW_LISTEN_HOST  = "0.0.0.0"
MGW_LISTEN_PORT  = 10025
MAIL_SERVER_HOST = "127.0.0.1"
MAIL_SERVER_PORT = 10026
RISK_THRESHOLD   = 0.70

# ─── Logging ─────────────────────────────────────────────────────────────────
# os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE),
               logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mail_filter")


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic module importer
# ══════════════════════════════════════════════════════════════════════════════
def _import_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A — Semantic Preprocessing
# Extracts phishing signals WITHOUT destroying them
# Returns metadata dict passed to classifiers alongside clean text
# ══════════════════════════════════════════════════════════════════════════════
def semantic_track(msg, raw_body: str) -> dict:
    """
    Preserve and COUNT all semantic signals that normalization would destroy.
    These become metadata features for the classifiers.
    """
    subject  = msg.get("Subject", "") or ""
    text     = raw_body
    lower    = text.lower()
    subj_low = subject.lower()

    # ── Hidden link detection ─────────────────────────────────────────────────
    url_pattern = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s<>"]*)?', re.I)
    href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.I)
    all_urls  = url_pattern.findall(text)
    href_urls = href_pattern.findall(text)

    # Find mismatched display text vs href (hidden redirect)
    mismatch_pattern = re.compile(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', re.I)
    mismatches = 0
    for m in mismatch_pattern.finditer(text):
        href_url, display = m.group(1), m.group(2).strip()
        if display.startswith("http") and display not in href_url:
            mismatches += 1

    # ── IP address in URLs ────────────────────────────────────────────────────
    ip_in_url = sum(1 for u in all_urls
                    if re.search(r'https?://\d+\.\d+\.\d+\.\d+', u))

    # ── Percent-encoded / obfuscated URLs ─────────────────────────────────────
    encoded_urls = sum(1 for u in all_urls if '%' in u)

    # ── Suspicious TLDs ───────────────────────────────────────────────────────
    bad_tlds = {'.tk','.ml','.ga','.cf','.gq','.xyz','.top',
                '.club','.online','.site','.work','.date','.loan'}
    suspicious_tld_count = sum(
        1 for u in all_urls
        if any(u.lower().split('/')[2].endswith(t) for t in bad_tlds
               if '/' in u and len(u.split('/')) > 2))

    # ── Embedded code detection ───────────────────────────────────────────────
    has_script     = int(bool(re.search(r'<script', text, re.I)))
    has_onclick    = int(bool(re.search(r'onclick\s*=', text, re.I)))
    has_onload     = int(bool(re.search(r'onload\s*=', text, re.I)))
    has_iframe     = int(bool(re.search(r'<iframe', text, re.I)))
    has_base64     = int(bool(re.search(r'base64[,\s]', text, re.I)))
    has_data_uri   = int(bool(re.search(r'data:[^;]+;base64', text, re.I)))
    has_eval       = int(bool(re.search(r'\beval\s*\(', text, re.I)))
    has_unescape   = int(bool(re.search(r'unescape\s*\(', text, re.I)))
    has_form       = int(bool(re.search(r'<form', text, re.I)))
    has_input_pass = int(bool(re.search(r'<input[^>]+type=["\']?password', text, re.I)))

    # ── HTML entity obfuscation ───────────────────────────────────────────────
    html_entity_count = len(re.findall(r'&[#a-zA-Z0-9]+;', text))
    decoded = html.unescape(text)
    obfuscated_chars  = int(decoded != text)

    # ── Phishing keyword counts (PRESERVED, not removed) ─────────────────────
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
    keyword_counts = {
        f"kw_{k.replace(' ','_')}": lower.count(k)
        for k in PHISHING_KEYWORDS
    }
    total_phishing_kw  = sum(keyword_counts.values())
    unique_phishing_kw = sum(1 for v in keyword_counts.values() if v > 0)

    # ── Urgency / Fear / Curiosity composite scores ───────────────────────────
    urgency_score = min(1.0, sum(lower.count(w) for w in [
        "urgent","immediately","asap","deadline","expire",
        "limited time","action required","hours remaining"
    ]) * 0.2)

    fear_score = min(1.0, sum(lower.count(w) for w in [
        "suspended","terminated","closed","blocked","restricted",
        "unauthorized","fraud","compromised","hacked"
    ]) * 0.2)

    curiosity_score = min(1.0, sum(lower.count(w) for w in [
        "winner","won","prize","selected","chosen","lucky","claim",
        "congratulations","inheritance","lottery"
    ]) * 0.2)

    # ── Subject-level semantic signals ───────────────────────────────────────
    subject_has_urgent  = int(bool(re.search(r'urgent|immediate|asap', subj_low)))
    subject_has_verify  = int(bool(re.search(r'verify|confirm|validate', subj_low)))
    subject_has_alert   = int("alert" in subj_low)
    subject_all_caps    = int(subject.isupper() and len(subject) > 3)
    subject_caps_ratio  = (sum(1 for c in subject if c.isupper())
                           / max(len(subject), 1))
    subject_money       = int(any(s in subject for s in ["$","€","£","¥","money","wire"]))
    subject_exclamation = subject.count("!")
    subject_has_numbers = int(any(c.isdigit() for c in subject))
    subject_has_special = int(any(c in "!@#$%^&*()" for c in subject))

    # ── Authentication / header trust signals ─────────────────────────────────
    auth_results = msg.get("Authentication-Results","") or ""
    auth_lower   = auth_results.lower()
    spf_fail     = int("spf=fail"   in auth_lower)
    dkim_fail    = int("dkim=fail"  in auth_lower)
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
        bool(reply_to) and reply_domain != from_domain or
        bool(return_path) and return_domain != from_domain
    )

    suspicious_tld_sender = int(any(from_domain.endswith(t) for t in bad_tlds))
    has_numeric_in_domain = int(any(c.isdigit() for c in from_domain))

    received_hops = len(msg.get_all("Received", []))
    date_str      = msg.get("Date","") or ""
    date_is_future = 0
    if date_str:
        try:
            from email.utils import parsedate_to_datetime
            date_obj = parsedate_to_datetime(date_str)
            date_is_future = int(date_obj > datetime.now(date_obj.tzinfo))
        except Exception:
            pass

    # ── Money / financial signals ─────────────────────────────────────────────
    dollar_count = text.count("$")
    total_money  = dollar_count + text.count("€") + text.count("£")

    # ── Assemble full semantic metadata ───────────────────────────────────────
    semantic = {
        # URL analysis
        "url_count":             len(all_urls),
        "href_url_count":        len(href_urls),
        "url_mismatch_count":    mismatches,
        "url_has_ip":            int(ip_in_url > 0),
        "url_encoded_count":     encoded_urls,
        "url_suspicious_tlds":   suspicious_tld_count,

        # Embedded code
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

        # Obfuscation
        "html_entity_count":     html_entity_count,
        "obfuscated_chars":      obfuscated_chars,

        # Keywords
        **keyword_counts,
        "total_phishing_keywords":  total_phishing_kw,
        "unique_phishing_keywords": unique_phishing_kw,

        # Composite
        "urgency_score":         urgency_score,
        "fear_score":            fear_score,
        "curiosity_score":       curiosity_score,

        # Subject
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

        # Auth / sender
        "spf_fail":              spf_fail,
        "dkim_fail":             dkim_fail,
        "dmarc_fail":            dmarc_fail,
        "has_dkim":              has_dkim,
        "domain_mismatch":       domain_mismatch,
        "suspicious_tld_sender": suspicious_tld_sender,
        "has_numeric_in_domain": has_numeric_in_domain,
        "received_hops":         received_hops,
        "date_is_future":        date_is_future,

        # Financial
        "dollar_count":          dollar_count,
        "total_money_symbols":   total_money,

        # Misc
        "has_from":     int(bool(msg.get("From"))),
        "has_to":       int(bool(msg.get("To"))),
        "has_cc":       int(bool(msg.get("Cc"))),
        "has_bcc":      int(bool(msg.get("Bcc"))),
        "has_subject":  int(bool(msg.get("Subject"))),
        "has_date":     int(bool(msg.get("Date"))),
        "has_message_id": int(bool(msg.get("Message-ID"))),
        "has_reply_to": int(bool(reply_to)),
        "has_return_path": int(bool(return_path)),
    }
    return semantic


# ══════════════════════════════════════════════════════════════════════════════
# TRACK B — Structural Preprocessing
# Cleans and tokenizes text for ML/NLP model input
# ══════════════════════════════════════════════════════════════════════════════
def structural_track(raw_body: str) -> str:
    """
    Prepare clean text for NLP/ML models.
    Remove markup but keep semantic-preserving placeholders.
    """
    text = raw_body

    # Replace URLs with placeholder (preserves presence signal)
    text = re.sub(r'https?://\S+', ' URL_TOKEN ', text)

    # Replace email addresses with placeholder
    text = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', ' EMAIL_TOKEN ', text)

    # Replace IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' IP_TOKEN ', text)

    # Decode HTML entities
    text = html.unescape(text)

    # Strip HTML tags (after extracting signals in semantic track)
    text = re.sub(r'<[^>]+>', ' ', text)

    # Replace money amounts
    text = re.sub(r'[\$€£¥]\s*[\d,]+', ' MONEY_TOKEN ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Lowercase
    text = text.lower()

    return text


# ══════════════════════════════════════════════════════════════════════════════
# Body extractor — returns raw HTML-inclusive body for both tracks
# ══════════════════════════════════════════════════════════════════════════════
def extract_body(msg) -> str:
    """Extract full body including HTML parts for semantic analysis."""
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
        return {"risk_probability": 0.5, "risk_factors": [str(exc)]}


def analyze_body(clean_text: str, semantic_meta: dict) -> dict:
    try:
        mod = _import_script(BODY_SCRIPT, "body_analyzer")
        return mod.analyze(clean_text, semantic_meta)
    except Exception as exc:
        logger.error(f"Body analysis failed: {exc}")
        return {"risk_probability": 0.5, "risk_factors": [str(exc)]}


# ══════════════════════════════════════════════════════════════════════════════
# Verdict logger
# ══════════════════════════════════════════════════════════════════════════════
def log_verdict(meta, header_r, body_r):
    ts    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    h_p   = header_r.get("risk_probability", 0.0)
    b_p   = body_r.get("risk_probability",   0.0)
    block = (
        "\n---------NEW Email message------------------------\n"
        f"[{ts}] {meta['message_id']}\n"
        f"from: {meta['from']} -> to: {meta['to']}\n"
        f"Header Analyzer = [{h_p:.4f}]\n"
        f"Body Analyzer   = [{b_p:.4f}]\n"
        "------Email Analyzer DONE------------------------"
    )
    logger.info(block)


# ══════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ══════════════════════════════════════════════════════════════════════════════
def process_message(raw_bytes: bytes):
    msg = message_from_bytes(raw_bytes)

    meta = {
        "message_id": msg.get("Message-ID", f"<gen-{time.time()}@mgw>"),
        "from":       msg.get("From",       "unknown"),
        "to":         msg.get("To",         "unknown"),
    }

    # Extract raw body (HTML preserved for semantic track)
    raw_body = extract_body(msg)

    # ── Track A: Semantic ─────────────────────────────────────────────────────
    semantic_meta = semantic_track(msg, raw_body)

    # ── Track B: Structural ───────────────────────────────────────────────────
    clean_text = structural_track(raw_body)

    # ── Models ────────────────────────────────────────────────────────────────
    header_result = analyze_header(semantic_meta)
    body_result   = analyze_body(clean_text, semantic_meta)

    # ── Log ───────────────────────────────────────────────────────────────────
    log_verdict(meta, header_result, body_result)

    # ── Decision ──────────────────────────────────────────────────────────────
    h_prob = header_result.get("risk_probability", 0.0)
    b_prob = body_result.get("risk_probability",   0.0)
    risk   = max(h_prob, b_prob)

    if risk >= RISK_THRESHOLD:
        logger.warning(
            f"[DROPPED] {meta['message_id']} risk={risk:.4f} >= {RISK_THRESHOLD}"
        )
    else:
        logger.info(
            f"[PASS] {meta['message_id']} risk={risk:.4f} < {RISK_THRESHOLD}"
        )
        _forward(raw_bytes, meta["from"], meta["to"])


def _forward(raw_bytes, from_addr, to_addr):
    try:
        with smtplib.SMTP(MAIL_SERVER_HOST, MAIL_SERVER_PORT, timeout=30) as s:
            s.sendmail(from_addr, [to_addr], raw_bytes)
        logger.info(f"Email forwarded to {to_addr}")
    except Exception as exc:
        logger.error(f"Forwarding failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# SMTP proxy server
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
                self._data += ((line[1:] if line.startswith(".") else line) + "\r\n").encode()
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
    logger.info(f"MailFilter SMTP proxy listening on {MGW_LISTEN_HOST}:{MGW_LISTEN_PORT}")
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

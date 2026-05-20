#!/usr/bin/env python3
"""
~/MGW/mail_filter.py
Mail Gateway Filter — Main Controller

Multi-layer defence pipeline:
  Track A (Semantic)    — counts/preserves phishing signals as metadata
  Track B (Structural)  — cleans & tokenises text for ML/NLP models
  Track C (Attachment)  — classifies attachments, dispatches to sandbox API
"""
from __future__ import annotations
import sys, os, json, logging, asyncio, time, re, html, importlib.util
from datetime import datetime
from email import message_from_bytes
from email.policy import default as email_policy
import smtplib

# ─── Paths ────────────────────────────────────────────────────────────────────
MGW_ROOT        = os.path.expanduser("~/MGW")
HEADER_SCRIPT   = os.path.join(MGW_ROOT, "models", "Header",  "header.py")
BODY_SCRIPT     = os.path.join(MGW_ROOT, "models", "Body",    "body.py")
SANDBOX_SCRIPT  = os.path.join(MGW_ROOT, "models", "Sandbox", "sandbox_client.py")
LOG_FILE        = os.path.join(MGW_ROOT, "mail_filter.log")
SANDBOX_LOG     = os.path.join(MGW_ROOT, "sandbox.log")

# ─── SMTP settings ────────────────────────────────────────────────────────────
MGW_LISTEN_HOST  = "0.0.0.0"
MGW_LISTEN_PORT  = 10025
MAIL_SERVER_HOST = "127.0.0.1"
MAIL_SERVER_PORT = 10026
RISK_THRESHOLD   = 0.70          # overall combined threshold

# ─── Model-selection resource thresholds ─────────────────────────────────────
CPU_THRESHOLD_PCT  = 75.0        # if CPU  >  this → use TF-IDF fallback
RAM_THRESHOLD_PCT  = 80.0        # if RAM  >  this → use TF-IDF fallback

# ─── Score weights (final fusion) ────────────────────────────────────────────
W_HEADER     = 0.30
W_BODY       = 0.40
W_ATTACHMENT = 0.30

# ─── Sandbox API endpoints (set to real IPs when sandboxes are live) ──────────
SANDBOX_WIN_URL   = os.environ.get("SANDBOX_WIN_URL",   "http://192.168.200.10:8080")
SANDBOX_LIN_URL   = os.environ.get("SANDBOX_LIN_URL",   "http://192.168.200.20:8080")
SANDBOX_API_TOKEN = os.environ.get("SANDBOX_API_TOKEN", "changeme")

# ─── Attachment classification maps ──────────────────────────────────────────
WIN_ONLY_EXTS  = {".exe",".msi",".bat",".cmd",".ps1",".vbs",".lnk"}
LIN_ONLY_EXTS  = {".sh",".elf",".run",".deb",".rpm",".bin"}
BOTH_EXTS      = {
    ".js",".zip",".7z",".rar",".iso",".img",
    ".docm",".xlsm",".pptm",".rtf",".pdf",
    ".html",".htm",".link",
}
ARCHIVE_EXTS   = {".zip",".7z",".rar",".iso",".img"}
OFFICE_EXTS    = {".docm",".xlsm",".pptm",".rtf",".pdf"}
SCRIPT_EXTS    = {".js",".vbs",".ps1",".bat",".cmd",".sh"}

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
    """Return current CPU / RAM usage percentages (requires psutil)."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.3)
        ram = psutil.virtual_memory().percent
        return {"cpu_pct": cpu, "ram_pct": ram, "healthy": True}
    except ImportError:
        # psutil not available → assume resources OK
        return {"cpu_pct": 0.0, "ram_pct": 0.0, "healthy": False}


def _use_roberta(health: dict) -> bool:
    """Decide whether to use RoBERTa or TF-IDF based on current load."""
    if not health["healthy"]:
        return True          # can't measure → prefer best model
    overloaded = (
        health["cpu_pct"] > CPU_THRESHOLD_PCT or
        health["ram_pct"] > RAM_THRESHOLD_PCT
    )
    return not overloaded


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A — Semantic Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def semantic_track(msg, raw_body: str) -> dict:
    """
    Preserve and COUNT all semantic signals that normalisation would destroy.
    Returns a metadata dict forwarded to every classifier.
    """
    subject  = msg.get("Subject", "") or ""
    text     = raw_body
    lower    = text.lower()
    subj_low = subject.lower()

    # ── URLs ─────────────────────────────────────────────────────────────────
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

    ip_in_url       = sum(1 for u in all_urls
                          if re.search(r'https?://\d+\.\d+\.\d+\.\d+', u))
    encoded_urls    = sum(1 for u in all_urls if '%' in u)
    bad_tlds        = {'.tk','.ml','.ga','.cf','.gq','.xyz','.top',
                       '.club','.online','.site','.work','.date','.loan'}
    suspicious_tld_count = sum(
        1 for u in all_urls
        if '/' in u and len(u.split('/')) > 2
        and any(u.lower().split('/')[2].endswith(t) for t in bad_tlds))

    # ── Embedded code ─────────────────────────────────────────────────────────
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

    # ── Obfuscation ───────────────────────────────────────────────────────────
    html_entity_count = len(re.findall(r'&[#a-zA-Z0-9]+;', text))
    decoded           = html.unescape(text)
    obfuscated_chars  = int(decoded != text)

    # ── Phishing keywords ─────────────────────────────────────────────────────
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

    # ── Composite scores ──────────────────────────────────────────────────────
    urgency_score = min(1.0, sum(lower.count(w) for w in [
        "urgent","immediately","asap","deadline","expire",
        "limited time","action required","hours remaining"]) * 0.2)
    fear_score    = min(1.0, sum(lower.count(w) for w in [
        "suspended","terminated","closed","blocked","restricted",
        "unauthorized","fraud","compromised","hacked"]) * 0.2)
    curiosity_score = min(1.0, sum(lower.count(w) for w in [
        "winner","won","prize","selected","chosen","lucky","claim",
        "congratulations","inheritance","lottery"]) * 0.2)

    # ── Subject signals ───────────────────────────────────────────────────────
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

    # ── Authentication ────────────────────────────────────────────────────────
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

    semantic = {
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
        "subject_text":          subject,           # raw text for TF-IDF subject scoring
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
    return semantic


# ══════════════════════════════════════════════════════════════════════════════
# TRACK B — Structural Preprocessing (clean text for NLP)
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
# TRACK C — Attachment Analysis
# ══════════════════════════════════════════════════════════════════════════════
def _classify_attachment(filename: str, content_type: str) -> dict:
    """Determine attachment type and which sandbox(es) to use."""
    ext = os.path.splitext(filename.lower())[1] if filename else ""
    result = {
        "filename":     filename or "unknown",
        "extension":    ext,
        "content_type": content_type,
        "is_executable":    ext in WIN_ONLY_EXTS | LIN_ONLY_EXTS,
        "is_archive":       ext in ARCHIVE_EXTS,
        "is_office_macro":  ext in OFFICE_EXTS,
        "is_script":        ext in SCRIPT_EXTS,
        "is_shortcut":      ext in {".lnk", ".link"},
        "is_web":           ext in {".html", ".htm"},
        "sandbox_targets": [],
        "risk_base":        0.0,
    }
    if ext in WIN_ONLY_EXTS:
        result["sandbox_targets"] = ["windows"]
        result["risk_base"]       = 0.75
    elif ext in LIN_ONLY_EXTS:
        result["sandbox_targets"] = ["linux"]
        result["risk_base"]       = 0.65
    elif ext in BOTH_EXTS:
        result["sandbox_targets"] = ["windows", "linux"]
        result["risk_base"]       = 0.60
    elif ext:
        result["sandbox_targets"] = ["windows", "linux"]
        result["risk_base"]       = 0.40
    return result


def _dispatch_to_sandbox(payload_b64: str, meta: dict) -> dict:
    """
    Send attachment payload to the appropriate sandbox(es) via HTTP API.
    Returns sandbox verdict or a default if the sandbox is unreachable.
    """
    import urllib.request, base64, json as _json

    targets  = meta.get("sandbox_targets", [])
    verdicts = []
    urls_map = {"windows": SANDBOX_WIN_URL, "linux": SANDBOX_LIN_URL}

    for target in targets:
        url = urls_map.get(target)
        if not url:
            continue
        body_data = _json.dumps({
            "filename":     meta["filename"],
            "payload_b64":  payload_b64,
            "token":        SANDBOX_API_TOKEN,
        }).encode()
        endpoint = f"{url}/api/analyze"
        try:
            req = urllib.request.Request(
                endpoint,
                data=body_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                verdict = _json.loads(resp.read().decode())
                verdict["sandbox"] = target
                verdicts.append(verdict)
                sandbox_logger.info(
                    f"[{target.upper()} SANDBOX] file={meta['filename']} "
                    f"verdict={verdict.get('verdict','?')} "
                    f"risk={verdict.get('risk_score',0):.4f}"
                )
        except Exception as exc:
            sandbox_logger.warning(
                f"[{target.upper()} SANDBOX] unreachable for {meta['filename']}: {exc}"
            )
            verdicts.append({
                "sandbox":   target,
                "verdict":   "timeout",
                "risk_score": meta.get("risk_base", 0.5),
            })
    if not verdicts:
        return {"verdict": "no_sandbox", "risk_score": meta.get("risk_base", 0.5)}

    # Aggregate: worst-case risk from all sandbox verdicts
    max_risk = max(v.get("risk_score", 0) for v in verdicts)
    any_malicious = any(v.get("verdict") == "malicious" for v in verdicts)
    return {
        "verdict":   "malicious" if any_malicious else verdicts[0].get("verdict","unknown"),
        "risk_score": max_risk,
        "details":   verdicts,
    }


def attachment_track(msg) -> dict:
    """
    Extract all attachments, classify each, dispatch to sandbox, return aggregate result.
    """
    import base64

    attachment_results = []
    has_any = False

    for part in msg.walk():
        disp      = part.get_content_disposition() or ""
        ct        = part.get_content_type()
        filename  = part.get_filename() or ""

        # Treat inline images / application parts with filenames as attachments too
        if "attachment" in disp or (filename and ct != "text/plain"
                                     and ct != "text/html"
                                     and ct != "multipart/mixed"):
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            has_any = True
            meta = _classify_attachment(filename, ct)

            # URLs embedded as link attachments (.html, .htm, .link)
            if meta["is_web"] or meta["is_shortcut"]:
                # Treat content as text; scan for redirect URLs
                try:
                    text_content = payload.decode("utf-8", errors="replace")
                    urls = re.findall(r'https?://\S+', text_content)
                    meta["embedded_urls"] = urls[:20]
                except Exception:
                    meta["embedded_urls"] = []

            payload_b64 = base64.b64encode(payload).decode()
            verdict     = _dispatch_to_sandbox(payload_b64, meta)
            attachment_results.append({**meta, **verdict})

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
    """Extract full body including HTML parts for both tracks."""
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
# Verdict logger
# ══════════════════════════════════════════════════════════════════════════════
def log_verdict(meta: dict, header_r: dict, body_r: dict,
                attach_r: dict, combined: float, action: str):
    ts       = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    h_prob   = header_r.get("risk_probability", 0.0)
    b_prob   = body_r.get("risk_probability",   0.0)
    a_prob   = attach_r.get("risk_probability", 0.0)
    h_engine = header_r.get("engine", "?")
    b_engine = body_r.get("engine",  "?")

    block = (
        f"\n{'='*60}\n"
        f"[{ts}] Message-ID: {meta['message_id']}\n"
        f"  From : {meta['from']}\n"
        f"  To   : {meta['to']}\n"
        f"  ── Model Results ──────────────────────────────────────\n"
        f"  Header  [{h_engine:20s}] risk = {h_prob:.6f}\n"
        f"  Body    [{b_engine:20s}] risk = {b_prob:.6f}\n"
        f"  Attach  [sandbox             ] risk = {a_prob:.6f}  "
        f"(count={attach_r.get('attachment_count',0)})\n"
        f"  ── Combined Score ─────────────────────────────────────\n"
        f"  COMBINED                       risk = {combined:.6f}  "
        f"threshold={RISK_THRESHOLD}\n"
        f"  ACTION : {action}\n"
        f"{'='*60}"
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

    # ── Resource health check (model selection) ───────────────────────────────
    health      = _resource_health()
    use_roberta = _use_roberta(health)
    logger.info(
        f"Resource health — CPU={health['cpu_pct']:.1f}%  "
        f"RAM={health['ram_pct']:.1f}%  "
        f"→ body_engine={'roberta' if use_roberta else 'tfidf'}"
    )

    # ── Extract raw body ──────────────────────────────────────────────────────
    raw_body = extract_body(msg)

    # ── Track A: Semantic ─────────────────────────────────────────────────────
    semantic_meta = semantic_track(msg, raw_body)

    # ── Track B: Structural ───────────────────────────────────────────────────
    clean_text = structural_track(raw_body)

    # ── Track C: Attachments → Sandbox ────────────────────────────────────────
    attach_result = attachment_track(msg)
    if attach_result["has_attachments"]:
        logger.info(
            f"Attachments detected: {attach_result['attachment_count']}  "
            f"risk={attach_result['risk_probability']:.4f}"
        )

    # ── Model calls ───────────────────────────────────────────────────────────
    header_result = analyze_header(semantic_meta)
    body_result   = analyze_body(clean_text, semantic_meta, use_roberta=use_roberta)

    # ── Combined risk (weighted) ──────────────────────────────────────────────
    h_prob = header_result.get("risk_probability", 0.0)
    b_prob = body_result.get("risk_probability",   0.0)
    a_prob = attach_result.get("risk_probability", 0.0)

    if attach_result["has_attachments"]:
        combined = W_HEADER * h_prob + W_BODY * b_prob + W_ATTACHMENT * a_prob
    else:
        # Redistribute attachment weight
        combined = (W_HEADER / (W_HEADER + W_BODY)) * h_prob + \
                   (W_BODY   / (W_HEADER + W_BODY)) * b_prob

    combined = float(min(combined, 1.0))

    # ── Decision ──────────────────────────────────────────────────────────────
    if combined >= RISK_THRESHOLD:
        action = f"DROPPED (risk={combined:.4f} >= threshold={RISK_THRESHOLD})"
        logger.warning(f"[DROPPED] {meta['message_id']} combined_risk={combined:.4f}")
    else:
        action = f"PASSED  (risk={combined:.4f} < threshold={RISK_THRESHOLD})"
        logger.info(f"[PASS] {meta['message_id']} combined_risk={combined:.4f}")
        _forward(raw_bytes, meta["from"], meta["to"])

    # ── Log full verdict ──────────────────────────────────────────────────────
    log_verdict(meta, header_result, body_result, attach_result, combined, action)


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

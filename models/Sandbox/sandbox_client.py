#!/usr/bin/env python3
"""
~/MGW/models/Sandbox/sandbox_client.py
Sandbox API client — dispatches attachments to Windows/Linux sandboxes
and retrieves behaviour reports.

The sandbox server (separate VM) must expose:
  POST /api/analyze   → { filename, payload_b64, token } → { verdict, risk_score, behaviors }
  GET  /api/status    → { online: true }

This client handles:
  • Attachment-type classification and OS routing
  • API dispatch (with retry + timeout)
  • Behavior report parsing: redirects, payload delivery, credential harvesting,
    info gathering, exploitation, social engineering, psychological manipulation
  • Link/URL de-redirection investigation
  • Risk aggregation and return to mail_filter
"""
from __future__ import annotations
import os, sys, json, logging, base64, re, time
import urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

MGW_ROOT    = Path.home() / "MGW"
SANDBOX_DIR = MGW_ROOT / "models" / "Sandbox"
LOG_FILE    = SANDBOX_DIR / "sandbox_client.log"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

SANDBOX_WIN_URL   = os.environ.get("SANDBOX_WIN_URL",   "http://192.168.200.10:8080")
SANDBOX_LIN_URL   = os.environ.get("SANDBOX_LIN_URL",   "http://192.168.200.20:8080")
SANDBOX_API_TOKEN = os.environ.get("SANDBOX_API_TOKEN", "changeme")

RETRY_COUNT   = 2
RETRY_DELAY   = 3     # seconds
TIMEOUT       = 45    # seconds per request

logger = logging.getLogger("sandbox_client")
if not logger.handlers:
    h = logging.FileHandler(LOG_FILE)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(h)

# ─── Attachment classification ────────────────────────────────────────────────
WIN_ONLY_EXTS = {".exe",".msi",".bat",".cmd",".ps1",".vbs",".lnk"}
LIN_ONLY_EXTS = {".sh",".elf",".run",".deb",".rpm",".bin"}
BOTH_EXTS     = {
    ".js",".zip",".7z",".rar",".iso",".img",
    ".docm",".xlsm",".pptm",".rtf",".pdf",
    ".html",".htm",".link",
}

BEHAVIOR_RISK_SCORES = {
    "redirect":                  0.40,
    "payload_delivery":          0.85,
    "credential_harvesting":     0.90,
    "information_gathering":     0.65,
    "exploitation":              0.95,
    "buffer_overflow":           0.95,
    "remote_code_execution":     0.98,
    "social_engineering":        0.60,
    "psychological_manipulation":0.55,
    "file_creation":             0.50,
    "network_connection":        0.45,
    "process_injection":         0.90,
    "registry_modification":     0.70,
    "persistence":               0.80,
    "high_cpu_usage":            0.30,
    "high_ram_usage":            0.30,
    "install_detected":          0.70,
}


def classify_attachment(filename: str) -> dict:
    """Determine attachment type and which sandbox(es) to target."""
    ext = os.path.splitext(filename.lower())[1] if filename else ""
    info = {
        "filename":        filename,
        "extension":       ext,
        "sandbox_targets": [],
        "risk_base":       0.30,
        "is_executable":   ext in WIN_ONLY_EXTS | LIN_ONLY_EXTS,
        "is_archive":      ext in {".zip",".7z",".rar",".iso",".img"},
        "is_office_macro": ext in {".docm",".xlsm",".pptm",".rtf",".pdf"},
        "is_script":       ext in {".js",".vbs",".ps1",".bat",".cmd",".sh"},
        "is_shortcut":     ext in {".lnk",".link"},
        "is_web":          ext in {".html",".htm"},
    }
    if ext in WIN_ONLY_EXTS:
        info["sandbox_targets"] = ["windows"]
        info["risk_base"]       = 0.75
    elif ext in LIN_ONLY_EXTS:
        info["sandbox_targets"] = ["linux"]
        info["risk_base"]       = 0.65
    elif ext in BOTH_EXTS:
        info["sandbox_targets"] = ["windows", "linux"]
        info["risk_base"]       = 0.60
    else:
        # Unknown extension → scan both, conservative base risk
        info["sandbox_targets"] = ["windows", "linux"]
        info["risk_base"]       = 0.40
    return info


# ─── Redirect chain investigation ─────────────────────────────────────────────
def investigate_redirect(url: str, max_hops: int = 10) -> dict:
    """
    Follow redirect chain for a URL and assess each hop for suspicion.
    Returns risk score and hop list.
    """
    import urllib.request
    BAD_TLDS    = {".tk",".ml",".ga",".cf",".gq",".xyz",".top",
                   ".club",".online",".site",".work",".date",".loan"}
    SHORTENERS  = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly",
                   "buff.ly","rebrand.ly","cutt.ly"}

    hops        = []
    risk        = 0.0
    current_url = url

    for i in range(max_hops):
        try:
            req = urllib.request.Request(current_url,
                                         headers={"User-Agent": "Mozilla/5.0"},
                                         method="HEAD")
            # Do NOT follow redirects automatically
            class _NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None

            opener = urllib.request.build_opener(_NoRedirect)
            try:
                resp      = opener.open(req, timeout=10)
                final_url = resp.geturl()
                code      = resp.getcode()
            except urllib.error.HTTPError as e:
                code      = e.code
                final_url = e.headers.get("Location", current_url)

            domain = re.sub(r'https?://', '', final_url).split('/')[0].lower()
            flags  = []
            if any(domain.endswith(t) for t in BAD_TLDS):
                flags.append("suspicious_tld"); risk += 0.30
            if domain in SHORTENERS:
                flags.append("url_shortener");  risk += 0.15
            if re.search(r'\d+\.\d+\.\d+\.\d+', domain):
                flags.append("ip_address");     risk += 0.40
            if "login" in final_url.lower() or "signin" in final_url.lower():
                flags.append("login_page");     risk += 0.25
            if "account" in final_url.lower() or "password" in final_url.lower():
                flags.append("credential_page");risk += 0.30

            hops.append({
                "hop":    i + 1,
                "url":    final_url,
                "code":   code,
                "domain": domain,
                "flags":  flags,
            })
            if code not in (301, 302, 303, 307, 308):
                break
            current_url = final_url
        except Exception as exc:
            hops.append({"hop": i + 1, "url": current_url, "error": str(exc)})
            break

    risk = min(risk, 1.0)
    return {
        "original_url": url,
        "hop_count":    len(hops),
        "hops":         hops,
        "risk_score":   round(risk, 4),
        "verdict":      "suspicious" if risk >= 0.40 else "clean",
    }


# ─── Sandbox API call ─────────────────────────────────────────────────────────
def _api_call(endpoint: str, payload: dict, retries: int = RETRY_COUNT) -> dict | None:
    body = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        try:
            req  = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json",
                         "X-API-Token": SANDBOX_API_TOKEN},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            logger.warning(
                f"[API] {endpoint} attempt {attempt+1}/{retries+1} failed: {exc}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
    return None


# ─── Behavior report parser ───────────────────────────────────────────────────
def parse_behavior_report(report: dict) -> dict:
    """
    Map sandbox behavior tags to risk scores.
    Expected report format (from sandbox VM):
      {
        "verdict": "malicious"|"suspicious"|"clean",
        "behaviors": ["payload_delivery", "network_connection", ...],
        "os_stats": {"cpu_peak": 85, "ram_peak": 70, "new_files": 12},
        "risk_score": 0.0-1.0
      }
    """
    behaviors  = report.get("behaviors", [])
    os_stats   = report.get("os_stats",  {})
    raw_risk   = float(report.get("risk_score", 0.0))

    behavior_risk = 0.0
    matched       = []
    for beh in behaviors:
        score = BEHAVIOR_RISK_SCORES.get(beh.lower(), 0.20)
        behavior_risk = max(behavior_risk, score)
        matched.append(f"{beh}={score:.2f}")

    # OS resource anomalies
    resource_risk = 0.0
    cpu_peak = float(os_stats.get("cpu_peak", 0))
    ram_peak = float(os_stats.get("ram_peak", 0))
    new_files= int(os_stats.get("new_files", 0))
    if cpu_peak > 80:  resource_risk += 0.20
    if ram_peak > 80:  resource_risk += 0.20
    if new_files > 5:  resource_risk += 0.15
    resource_risk = min(resource_risk, 0.50)

    # Combine: highest behavior risk dominates, resource risk adds
    final_risk = min(max(raw_risk, behavior_risk) + resource_risk * 0.3, 1.0)

    return {
        "verdict":         report.get("verdict", "unknown"),
        "behaviors":       behaviors,
        "matched_risks":   matched,
        "behavior_risk":   round(behavior_risk, 4),
        "resource_risk":   round(resource_risk, 4),
        "final_risk":      round(final_risk, 4),
        "os_stats":        os_stats,
    }


# ─── Main dispatch ────────────────────────────────────────────────────────────
def analyze_attachment(filename: str, payload_bytes: bytes) -> dict:
    """
    Full attachment analysis pipeline:
      1. Classify attachment → determine sandbox targets
      2. Dispatch to sandbox(es)
      3. Parse behaviour reports
      4. Investigate embedded redirect URLs (for .html / .link / .htm)
      5. Return aggregate risk
    """
    info = classify_attachment(filename)
    logger.info(
        f"Attachment: {filename}  ext={info['extension']}  "
        f"targets={info['sandbox_targets']}  base_risk={info['risk_base']}"
    )

    payload_b64 = base64.b64encode(payload_bytes).decode()
    verdicts    = []
    urls_map    = {"windows": SANDBOX_WIN_URL, "linux": SANDBOX_LIN_URL}

    for target in info["sandbox_targets"]:
        url      = urls_map.get(target)
        endpoint = f"{url}/api/analyze"
        resp     = _api_call(endpoint, {
            "filename":    filename,
            "payload_b64": payload_b64,
            "token":       SANDBOX_API_TOKEN,
        })
        if resp:
            parsed = parse_behavior_report(resp)
            parsed["sandbox"] = target
            verdicts.append(parsed)
            logger.info(
                f"[{target.upper()}] verdict={parsed['verdict']}  "
                f"risk={parsed['final_risk']:.4f}  behaviors={parsed['behaviors']}"
            )
        else:
            # Sandbox timeout → use base risk as conservative estimate
            logger.warning(f"[{target.upper()}] sandbox timeout for {filename}")
            verdicts.append({
                "sandbox":    target,
                "verdict":    "timeout",
                "final_risk": info["risk_base"],
                "behaviors":  [],
            })

    # ── Link / web attachment: investigate redirect chains ────────────────────
    redirect_results = []
    if info["is_web"] or info["is_shortcut"]:
        try:
            text    = payload_bytes.decode("utf-8", errors="replace")
            urls    = re.findall(r'https?://[^\s"\'<>]+', text)[:10]
            for u in urls:
                rd = investigate_redirect(u)
                redirect_results.append(rd)
                logger.info(
                    f"[REDIRECT] {u} → hops={rd['hop_count']} "
                    f"risk={rd['risk_score']:.4f} verdict={rd['verdict']}"
                )
        except Exception as exc:
            logger.warning(f"Redirect investigation failed: {exc}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_risks = [v.get("final_risk", 0) for v in verdicts]
    if redirect_results:
        all_risks += [r.get("risk_score", 0) for r in redirect_results]

    max_risk      = max(all_risks) if all_risks else info["risk_base"]
    any_malicious = any(v.get("verdict") == "malicious" for v in verdicts)

    result = {
        "filename":         filename,
        "classification":   info,
        "sandbox_verdicts": verdicts,
        "redirect_chain":   redirect_results,
        "verdict":          "malicious" if any_malicious else
                            ("suspicious" if max_risk >= 0.50 else "clean"),
        "risk_score":       round(max_risk, 6),
        "timestamp":        datetime.utcnow().isoformat(),
    }

    logger.info(
        f"[FINAL] {filename} verdict={result['verdict']} risk={result['risk_score']:.4f}"
    )
    return result


# ─── Standalone usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sandbox attachment analyser")
    ap.add_argument("file", help="Path to attachment file")
    args = ap.parse_args()

    fpath = Path(args.file)
    if not fpath.exists():
        print(f"File not found: {fpath}", file=sys.stderr)
        sys.exit(1)

    with open(fpath, "rb") as f:
        data = f.read()

    result = analyze_attachment(fpath.name, data)
    print(json.dumps(result, indent=2))

#!/usr/bin/env python3
"""
~/MGW/models/Attachment/sandbox_client.py
CAPEv2-integrated sandbox client.

Submits attachments directly to the LOCAL CAPEv2 REST API on MGW (127.0.0.1:8000).
CAPE master routes to CAPE_Linux or CAPE_WIN automatically via platform= tag.
Saves per-attachment analysis detail to ~/MGW/Attachment/<timestamp>_<filename>.json
"""
from __future__ import annotations
import os, sys, json, logging, base64, re, time, tempfile, pathlib, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

# ─── Paths ───────────────────────────────────────────────────────────────────
MGW_ROOT       = Path.home() / "MGW"
SANDBOX_DIR    = MGW_ROOT / "models" / "Attachment
ATTACHMENT_DIR = MGW_ROOT / "Attachment"
LOG_FILE       = SANDBOX_DIR / "sandbox_client.log"

SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CAPEv2 local API ────────────────────────────────────────────────────────
CAPE_API  = os.environ.get("CAPE_API", "http://127.0.0.1:8000")
TIMEOUT   = int(os.environ.get("CAPE_TIMEOUT", "300"))   # max wait for report
POLL_INT  = int(os.environ.get("CAPE_POLL",    "10"))    # polling interval

# ─── Extension → platform routing ────────────────────────────────────────────
WIN_ONLY_EXTS = {".exe", ".msi", ".bat", ".cmd", ".ps1", ".vbs", ".lnk"}
LIN_ONLY_EXTS = {".sh",  ".elf", ".run", ".deb", ".rpm", ".bin"}
BOTH_EXTS     = {
    ".js",   ".zip",  ".7z",  ".rar", ".iso", ".img",
    ".docm", ".xlsm", ".pptm", ".rtf", ".pdf",
    ".html", ".htm",  ".link",
}

BEHAVIOR_RISK_SCORES = {
    "redirect":                   0.40,
    "payload_delivery":           0.85,
    "credential_harvesting":      0.90,
    "information_gathering":      0.65,
    "exploitation":               0.95,
    "buffer_overflow":            0.95,
    "remote_code_execution":      0.98,
    "social_engineering":         0.60,
    "psychological_manipulation": 0.55,
    "file_creation":              0.50,
    "network_connection":         0.45,
    "process_injection":          0.90,
    "registry_modification":      0.70,
    "persistence":                0.80,
}

# ─── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("sandbox_client")
if not logger.handlers:
    h = logging.FileHandler(LOG_FILE)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(h)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def _get_platforms(filename: str) -> list[str]:
    ext = os.path.splitext(filename.lower())[1]
    if ext in WIN_ONLY_EXTS:  return ["windows"]
    if ext in LIN_ONLY_EXTS:  return ["linux"]
    if ext in BOTH_EXTS:      return ["windows", "linux"]
    return ["windows", "linux"]


def _cape_post(path: str, data: dict | None = None,
               files: dict | None = None) -> dict | None:
    """Simple HTTP helper for CAPEv2 REST API."""
    url = f"{CAPE_API}{path}"
    try:
        if files:
            # multipart/form-data via temp file approach
            import urllib.request, io, mimetypes
            boundary = "---CAPEBoundary7f3d9a"
            body = b""
            for field, value in (data or {}).items():
                body += (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode()
            for field, (fname, fbytes) in files.items():
                body += (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field}"; filename="{fname}"\r\n'
                    f"Content-Type: application/octet-stream\r\n\r\n"
                ).encode() + fbytes + b"\r\n"
            body += f"--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(data or {}).encode() if data else None,
                headers={"Content-Type": "application/json"},
            )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning(f"CAPE API {path} failed: {exc}")
        return None


def _submit_task(filename: str, payload_bytes: bytes, platform: str) -> int | None:
    """Submit file to CAPEv2 API. Returns task_id or None."""
    result = _cape_post(
        "/apiv2/tasks/create/file/",
        data={"platform": platform, "timeout": 120, "enforce_timeout": "true"},
        files={"file": (filename, payload_bytes)},
    )
    if not result:
        return None
    tid = (result.get("data") or {}).get("task_ids", [None])[0] \
          or (result.get("data") or {}).get("task_id")
    logger.info(f"Submitted {filename} ({platform}) → task_id={tid}")
    return tid


def _wait_for_report(task_id: int) -> dict | None:
    """Poll CAPE until task is reported. Returns raw report or None on timeout."""
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INT)
        try:
            st = _cape_post.__func__ if False else None  # dummy
            r = _cape_get(f"/apiv2/tasks/status/{task_id}/")
            status = (r.get("data") if r else None)
            logger.info(f"Task {task_id} status: {status}")
            if status == "reported":
                return _cape_get(f"/apiv2/tasks/get/report/{task_id}/")
            if status in ("failed_analysis", "failed_processing"):
                logger.error(f"Task {task_id} failed: {status}")
                return None
        except Exception as exc:
            logger.warning(f"Poll error task {task_id}: {exc}")
    logger.error(f"Task {task_id} timed out after {TIMEOUT}s")
    return None


def _cape_get(path: str) -> dict | None:
    url = f"{CAPE_API}{path}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning(f"CAPE GET {path} failed: {exc}")
        return None


def _parse_report(raw: dict) -> dict:
    """Extract verdict and behaviors from raw CAPEv2 report."""
    report    = (raw.get("data") or raw) if raw else {}
    sigs      = report.get("signatures", [])
    malscore  = float(report.get("malscore", 0))
    behaviors = [s.get("name", "") for s in sigs[:20]]
    risk      = round(min(malscore / 10.0, 1.0), 4)

    # Enrich with behavior risk scores
    behavior_risk = 0.0
    for beh in behaviors:
        score = BEHAVIOR_RISK_SCORES.get(beh.lower(), 0.0)
        behavior_risk = max(behavior_risk, score)
    final_risk = round(min(max(risk, behavior_risk), 1.0), 4)

    verdict = (
        "malicious"  if final_risk > 0.70 else
        "suspicious" if final_risk > 0.40 else
        "clean"
    )

    info    = report.get("info", {})
    machine = info.get("machine", {})

    return {
        "verdict":       verdict,
        "risk_score":    final_risk,
        "malscore":      malscore,
        "behaviors":     behaviors,
        "behavior_risk": behavior_risk,
        "machine":       machine.get("name", "?"),
        "platform":      machine.get("platform", "?"),
        "duration":      info.get("duration", 0),
        "task_id":       info.get("id"),
        "signatures":    [{"name": s.get("name"), "severity": s.get("severity")}
                          for s in sigs[:10]],
        "network": {
            "hosts":   [h.get("ip")   for h in report.get("network", {}).get("hosts",   [])[:10]],
            "domains": [d.get("domain") for d in report.get("network", {}).get("domains", [])[:10]],
        },
        "dropped_files": len(report.get("dropped", [])),
        "processes":     [p.get("process_name") for p in
                          report.get("behavior", {}).get("processes", [])[:10]],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Redirect chain investigation (kept from original)
# ═════════════════════════════════════════════════════════════════════════════
def investigate_redirect(url: str, max_hops: int = 10) -> dict:
    BAD_TLDS   = {".tk",".ml",".ga",".cf",".gq",".xyz",".top",
                  ".club",".online",".site",".work",".date",".loan"}
    SHORTENERS = {"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly",
                  "buff.ly","rebrand.ly","cutt.ly"}
    hops        = []
    risk        = 0.0
    current_url = url

    for i in range(max_hops):
        try:
            req = urllib.request.Request(
                current_url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
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
            hops.append({"hop": i+1, "url": final_url,
                         "code": code, "domain": domain, "flags": flags})
            if code not in (301, 302, 303, 307, 308):
                break
            current_url = final_url
        except Exception as exc:
            hops.append({"hop": i+1, "url": current_url, "error": str(exc)})
            break

    risk = min(risk, 1.0)
    return {
        "original_url": url,
        "hop_count":    len(hops),
        "hops":         hops,
        "risk_score":   round(risk, 4),
        "verdict":      "suspicious" if risk >= 0.40 else "clean",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Attachment log saver
# ═════════════════════════════════════════════════════════════════════════════
def _save_attachment_log(filename: str, analysis: dict) -> str:
    """Save detailed analysis JSON to ~/MGW/Attachment/. Returns saved path."""
    ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe    = re.sub(r'[^\w.\-]', '_', filename)[:60]
    outpath = ATTACHMENT_DIR / f"{ts}_{safe}.json"
    try:
        outpath.write_text(json.dumps(analysis, indent=2, default=str))
        logger.info(f"Attachment log saved: {outpath}")
    except Exception as exc:
        logger.error(f"Failed to save attachment log: {exc}")
    return str(outpath)


# ═════════════════════════════════════════════════════════════════════════════
# Main public interface
# ═════════════════════════════════════════════════════════════════════════════
def analyze_attachment(filename: str, payload_bytes: bytes) -> dict:
    """
    Full pipeline:
      1. Classify attachment → platforms
      2. Submit to CAPEv2 API for each platform
      3. Poll until reported
      4. Parse report
      5. Investigate redirect chains for web/link files
      6. Save detailed log to ~/MGW/Attachment/
      7. Return aggregate result to mail_filter
    """
    ext       = os.path.splitext(filename.lower())[1]
    platforms = _get_platforms(filename)

    logger.info(f"Analyzing: {filename}  ext={ext}  platforms={platforms}")

    cape_verdicts    = []
    redirect_results = []

    for platform in platforms:
        task_id = _submit_task(filename, payload_bytes, platform)
        if task_id is None:
            logger.warning(f"Failed to submit {filename} to CAPE ({platform})")
            cape_verdicts.append({
                "platform": platform, "verdict": "error",
                "risk_score": 0.50, "error": "submission_failed",
            })
            continue

        raw_report = _wait_for_report(task_id)
        if raw_report is None:
            cape_verdicts.append({
                "platform": platform, "verdict": "timeout",
                "risk_score": 0.50, "task_id": task_id,
            })
            continue

        parsed = _parse_report(raw_report)
        parsed["platform"] = platform
        cape_verdicts.append(parsed)
        logger.info(
            f"[{platform.upper()}] verdict={parsed['verdict']} "
            f"risk={parsed['risk_score']} task_id={parsed.get('task_id')}"
        )

    # ── Redirect investigation for web/link attachments ───────────────────────
    if ext in {".html", ".htm", ".link"}:
        try:
            text = payload_bytes.decode("utf-8", errors="replace")
            urls = re.findall(r'https?://[^\s"\'<>]+', text)[:10]
            for u in urls:
                rd = investigate_redirect(u)
                redirect_results.append(rd)
                logger.info(
                    f"[REDIRECT] {u} hops={rd['hop_count']} "
                    f"risk={rd['risk_score']} verdict={rd['verdict']}"
                )
        except Exception as exc:
            logger.warning(f"Redirect investigation error: {exc}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_risks = [v.get("risk_score", 0) for v in cape_verdicts]
    if redirect_results:
        all_risks += [r.get("risk_score", 0) for r in redirect_results]

    max_risk      = max(all_risks) if all_risks else 0.40
    any_malicious = any(v.get("verdict") == "malicious" for v in cape_verdicts)
    any_suspicious = any(v.get("verdict") == "suspicious" for v in cape_verdicts)

    final_verdict = (
        "malicious"  if any_malicious  else
        "suspicious" if any_suspicious or max_risk >= 0.50 else
        "clean"
    )

    analysis = {
        "filename":       filename,
        "extension":      ext,
        "platforms":      platforms,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "cape_verdicts":  cape_verdicts,
        "redirect_chain": redirect_results,
        "verdict":        final_verdict,
        "risk_score":     round(max_risk, 6),
    }

    # ── Save detailed log ─────────────────────────────────────────────────────
    log_path = _save_attachment_log(filename, analysis)
    analysis["log_path"] = log_path

    logger.info(
        f"[FINAL] {filename} verdict={final_verdict} "
        f"risk={max_risk:.4f} log={log_path}"
    )
    return analysis


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CAPEv2 attachment analyser")
    ap.add_argument("file", nargs="?", help="Path to attachment file")
    ap.add_argument("--status", action="store_true", help="Show CAPE machine status")
    args = ap.parse_args()

    if args.status:
        r = _cape_get("/apiv2/machines/list/")
        machines = (r.get("data") or []) if r else []
        print("=== CAPE Machines ===")
        for m in machines:
            print(f"  {m['name']:15} platform={m['platform']:8} status={m['status']}")
        sys.exit(0)

    if not args.file:
        ap.print_help()
        sys.exit(1)

    fpath = Path(args.file)
    if not fpath.exists():
        print(f"File not found: {fpath}", file=sys.stderr)
        sys.exit(1)

    result = analyze_attachment(fpath.name, fpath.read_bytes())
    print(json.dumps(result, indent=2, default=str))

#!/usr/bin/env python3
"""
~/MGW/models/Attachment/sandbox_client.py
CAPEv2 integration for MGW Mail Gateway

Called by mail_filter.py via analyze_attachment().

FIXES applied vs previous version
───────────────────────────────────
FIX-SC01: cape_verdicts now includes "behavior_risk" key (0-1 float) derived
          from malscore — decision_engine reads this and was getting 0.0 always.
FIX-SC02: cape_verdicts "behaviors" key now contains list of string labels
          (e.g. ["network_contact", "process_injection"]) rather than raw
          strace {ip, port} dicts. The strace data is preserved separately
          under "network_connections" for forensics but does NOT go into behaviors.
          This prevents decision_engine from calling dict.lower() and crashing.
FIX-SC03: Network connections from strace now also set cape_network_contact
          flag in the verdict via "network_connections" list so decision_engine
          can pick them up through the normal cape_network_contact signal.
"""

import os
import re
import time
import json
import logging
import hashlib
import mimetypes
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("/var/log/mgw")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] sandbox_client: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sandbox_client.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("sandbox_client")

# ---------------------------------------------------------------------------
# CAPEv2 connection settings
# ---------------------------------------------------------------------------
CAPE_HOST   = os.getenv("CAPE_HOST", "http://192.168.200.254")
CAPE_PORT   = int(os.getenv("CAPE_PORT", "8000"))
CAPE_APIKEY = os.getenv("CAPE_APIKEY", "")

CAPE_BASE   = f"{CAPE_HOST}:{CAPE_PORT}"
CAPE_FILES  = f"{CAPE_BASE}/apiv2/tasks/create/file/"
CAPE_REPORT = f"{CAPE_BASE}/apiv2/tasks/get/report/{{task_id}}/json"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
ANALYSIS_TIMEOUT = 120
POLL_INTERVAL    = 10
REQUEST_TIMEOUT  = 30
SUBMIT_RETRIES   = 3

# ---------------------------------------------------------------------------
# Package / platform mapping
# ---------------------------------------------------------------------------
CAPE_PACKAGE_MAP: Dict[str, str] = {
    ".exe": "exe", ".dll": "dll", ".doc": "doc", ".docx": "doc",
    ".xls": "xls", ".xlsx": "xls", ".xlsm": "xls", ".pdf": "pdf",
    ".js":  "js",  ".vbs": "vbs", ".ps1":  "ps1", ".bat": "exe",
    ".cmd": "exe", ".scr": "exe", ".msi":  "exe", ".zip": "zip",
    ".rar": "zip", ".7z":  "zip", ".elf":  "elf", ".sh":  "sh",
    ".py":  "py",
}

PLATFORM_MAP: Dict[str, List[str]] = {
    ".exe": ["windows"], ".dll": ["windows"], ".doc":  ["windows"],
    ".pdf": ["windows"], ".ps1": ["windows"], ".vbs":  ["windows"],
    ".elf": ["linux"],   ".sh":  ["linux"],   ".py":   ["linux"],
    ".zip": ["windows", "linux"],
}

FAILED_TASK_RISK = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _headers() -> dict:
    h = {"Accept": "application/json"}
    if CAPE_APIKEY:
        h["Authorization"] = f"Token {CAPE_APIKEY}"
    return h


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _resolve_package(filename: str) -> Optional[str]:
    return CAPE_PACKAGE_MAP.get(_get_extension(filename))


def _extract_task_id(response: requests.Response) -> Optional[int]:
    try:
        body = response.json()
        for key in ("task_id", "task_ids"):
            val = body.get(key) or (body.get("data") or {}).get(key)
            if val:
                return int(val[0] if isinstance(val, list) else val)
    except Exception:
        pass

    loc = response.headers.get("Location", "")
    m = re.search(r"/(\d+)/?$", loc)
    if m:
        return int(m.group(1))

    m = re.search(r'"task[_ ]?id"\s*:\s*(\d+)', response.text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def cape_is_alive() -> bool:
    try:
        r = requests.get(f"{CAPE_BASE}/apiv2/cuckoo/status/",
                         headers=_headers(), timeout=REQUEST_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Submit file to CAPE
# ---------------------------------------------------------------------------
def submit_file(file_data: bytes, filename: str) -> Optional[int]:
    if not file_data or len(file_data) == 0:
        log.error("Empty file rejected — payload is 0 bytes")
        return None

    pkg  = _resolve_package(filename) or "generic"
    ext  = _get_extension(filename)
    plat = PLATFORM_MAP.get(ext, ["windows"])[0]

    mime, _ = mimetypes.guess_type(filename)
    mime    = mime or "application/octet-stream"

    payload = {
        "package":   pkg,
        "timeout":   str(ANALYSIS_TIMEOUT),
        "options":   "procmemdump=1,screenshots=1",
        "platforms": json.dumps([{"platform": plat, "os_version": ""}]),
    }
    files = {"file": (filename, file_data, mime)}

    for attempt in range(1, SUBMIT_RETRIES + 1):
        try:
            r = requests.post(CAPE_FILES, headers=_headers(),
                              data=payload, files=files, timeout=REQUEST_TIMEOUT)
            if r.status_code in (200, 201):
                task_id = _extract_task_id(r)
                if task_id:
                    log.info(f"Submitted '{filename}' → Task ID: {task_id}")
                    return task_id
                log.warning(f"Submit OK but could not parse task_id from: {r.text[:200]}")
        except Exception as e:
            log.warning(f"Submit attempt {attempt} failed: {e}")

        if attempt < SUBMIT_RETRIES:
            time.sleep(2 ** attempt)

    log.error(f"All {SUBMIT_RETRIES} submit attempts failed for '{filename}'")
    return None


# ---------------------------------------------------------------------------
# Wait for task completion
# ---------------------------------------------------------------------------
def wait_for_task(task_id: int) -> Optional[str]:
    url      = f"{CAPE_BASE}/apiv2/tasks/get/status/{task_id}/"
    terminal = {"reported", "failed_analysis", "failed_processing",
                "failed_reporting", "aborted"}

    for _ in range(30):
        try:
            r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
            if r.ok:
                status = r.json().get("data", {}).get("status", "")
                log.info(f"Task {task_id} status: {status}")
                if status in terminal:
                    return status
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)

    log.warning(f"Task {task_id} did not reach terminal state within poll window")
    return None


# ---------------------------------------------------------------------------
# Get report
# ---------------------------------------------------------------------------
def get_report(task_id: int) -> Optional[Dict]:
    url = CAPE_REPORT.format(task_id=task_id)
    try:
        r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
        return r.json() if r.ok else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parse strace log for network connections
# FIX-SC03: Returns list of {ip, port} dicts for forensics only.
#           Callers must NOT put these into cape_verdicts["behaviors"].
# ---------------------------------------------------------------------------
def parse_strace_log(analysis_id: int) -> Dict:
    strace_path = (
        f"/home/rama/CAPEv2/storage/analyses/{analysis_id}/strace/strace.log"
    )
    connections: List[Dict] = []

    if os.path.exists(strace_path):
        try:
            with open(strace_path, "r", errors="replace") as f:
                for line in f:
                    if "connect" in line and "sin_port" in line:
                        ip_m   = re.search(r'inet_addr\("([^"]+)"\)', line)
                        port_m = re.search(r'htons\((\d+)\)', line)
                        if ip_m and port_m:
                            connections.append({
                                "ip":   ip_m.group(1),
                                "port": int(port_m.group(1)),
                            })
        except Exception as exc:
            log.warning(f"strace parse error for {analysis_id}: {exc}")

    return {
        "has_connections": len(connections) > 0,
        "connections":     connections,
    }


# ---------------------------------------------------------------------------
# Extract behavior labels from CAPE report
# FIX-SC02: Returns list of str labels — safe for decision_engine to iterate.
# ---------------------------------------------------------------------------
def _extract_behavior_labels(report: Dict) -> List[str]:
    """
    Parse CAPE report's behavior section into a list of string labels
    that decision_engine can safely iterate (calls .lower() on each).

    Known labels decision_engine checks: "process_injection"
    """
    labels: List[str] = []
    if not report:
        return labels

    behavior = report.get("behavior", {})

    # CAPE generic process behavior
    for proc in behavior.get("processes", []):
        for call in proc.get("calls", []):
            api = (call.get("api") or "").lower()
            if "inject" in api or "writeprocessmemory" in api:
                if "process_injection" not in labels:
                    labels.append("process_injection")
            if "createremotethread" in api:
                if "process_injection" not in labels:
                    labels.append("process_injection")

    # CAPE summary signatures
    for sig in report.get("signatures", []):
        name = (sig.get("name") or "").lower()
        if "inject" in name:
            if "process_injection" not in labels:
                labels.append("process_injection")
        if "network" in name or "connect" in name:
            if "network_contact" not in labels:
                labels.append("network_contact")
        if "drop" in name or "file_drop" in name:
            if "dropped_files" not in labels:
                labels.append("dropped_files")

    return labels


# ---------------------------------------------------------------------------
# Extract risk score from report
# ---------------------------------------------------------------------------
def extract_risk_score(report: Dict,
                       task_id: int,
                       task_status: Optional[str]) -> float:
    if task_status and task_status.startswith("failed"):
        return FAILED_TASK_RISK
    if not report:
        return FAILED_TASK_RISK

    malscore = report.get("malscore") or report.get("info", {}).get("score")
    if malscore is not None:
        score = float(malscore) / 10.0
        return min(max(score, 0.0), 1.0)

    return 0.3


# ---------------------------------------------------------------------------
# MAIN API — called by mail_filter.py
# ---------------------------------------------------------------------------
def analyze_attachment(filename: str,
                       file_data: bytes,
                       message_id: str = None) -> Dict[str, Any]:
    """
    Main entry point for mail_filter.py.

    Returns dict with keys:
      filename, verdict, risk_score, payload_size, task_id, task_status,
      cape_verdicts (list of dicts with string behaviors list),
      network_connections (strace data, forensics only)
    """
    log.info(
        f"analyze_attachment: '{filename}' ({len(file_data)} bytes) "
        f"msg={message_id}"
    )

    # Guard: empty payload
    if not file_data or len(file_data) == 0:
        return {
            "filename":            filename,
            "verdict":             "skipped",
            "risk_score":          0.05,
            "payload_size":        0,
            "cape_verdicts":       [],
            "network_connections": [],
        }

    # Submit to CAPE
    task_id = submit_file(file_data, filename)
    if not task_id:
        return {
            "filename":            filename,
            "verdict":             "submission_failed",
            "risk_score":          0.40,
            "payload_size":        len(file_data),
            "cape_verdicts":       [],
            "network_connections": [],
        }

    # Wait for completion
    task_status = wait_for_task(task_id)

    # Fetch report
    report = get_report(task_id)

    # Risk score
    risk_score = extract_risk_score(report, task_id, task_status)

    # FIX-SC02: behavior_labels are strings, safe for decision_engine
    behavior_labels = _extract_behavior_labels(report)

    # FIX-SC03: strace network connections — forensics, NOT in behaviors
    strace_data = parse_strace_log(task_id) if task_id else {"connections": []}

    # Merge network evidence into behavior labels
    if strace_data.get("has_connections") and "network_contact" not in behavior_labels:
        behavior_labels.append("network_contact")

    # FIX-SC01: behavior_risk derived from malscore so decision_engine gets it
    malscore_raw = 0.0
    if report:
        malscore_raw = float(report.get("malscore", 0) or 0)
    behavior_risk = min(1.0, malscore_raw / 10.0)

    # Network host count for cape_network signal
    network_hosts = strace_data.get("connections", [])

    # CAPE verdict entry — decision_engine reads: malscore, behavior_risk,
    # behaviors (list[str]), network.hosts, dropped_files
    ext  = _get_extension(filename)
    plat = PLATFORM_MAP.get(ext, ["windows"])[0]

    cape_verdict = {
        "platform":      plat,
        "task_id":       task_id,
        "malscore":      malscore_raw,                     # 0-10 raw
        "risk_score":    risk_score,                       # 0-1 normalised
        "behavior_risk": behavior_risk,                    # FIX-SC01: 0-1
        "behaviors":     behavior_labels,                  # FIX-SC02: list[str]
        "network": {
            "hosts": [c["ip"] for c in network_hosts],    # list[str] IPs
        },
        "dropped_files": len([
            b for b in (report.get("behavior", {})
                            .get("summary", {})
                            .get("files", [])) if True
        ]) if report else 0,
    }

    verdict = (
        "malicious"  if risk_score >= 0.7 else
        "suspicious" if risk_score >= 0.4 else
        "clean"
    )

    result = {
        "filename":            filename,
        "verdict":             verdict,
        "risk_score":          risk_score,
        "payload_size":        len(file_data),
        "task_id":             task_id,
        "task_status":         task_status,
        "cape_verdicts":       [cape_verdict],
        # Forensic strace data — kept separate from behaviors (FIX-SC03)
        "network_connections": strace_data.get("connections", []),
    }

    log.info(
        f"Result for '{filename}': verdict={verdict}  "
        f"risk={risk_score:.4f}  behaviors={behavior_labels}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sandbox_client.py <file_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as fh:
        data = fh.read()

    result = analyze_attachment(sys.argv[1], data)
    print(json.dumps(result, indent=2, default=str))

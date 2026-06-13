cat > /home/rama/MGW/models/Attachment/sandbox_client.py << 'EOF'
#!/usr/bin/env python3
"""
sandbox_client.py – MGW / CAPEv2 integration - CLEAN VERSION
"""

import os
import re
import time
import json
import logging
import hashlib
import mimetypes
import tempfile
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
CAPE_HOST = os.getenv("CAPE_HOST", "http://192.168.200.254")
CAPE_PORT = int(os.getenv("CAPE_PORT", "8000"))
CAPE_APIKEY = os.getenv("CAPE_APIKEY", "")

CAPE_BASE = f"{CAPE_HOST}:{CAPE_PORT}"
CAPE_FILES = f"{CAPE_BASE}/apiv2/tasks/create/file/"
CAPE_REPORT = f"{CAPE_BASE}/apiv2/tasks/get/report/{{task_id}}/json"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
ANALYSIS_TIMEOUT = 120
POLL_INTERVAL = 10
REQUEST_TIMEOUT = 30
SUBMIT_RETRIES = 3

# ---------------------------------------------------------------------------
# Package mapping
# ---------------------------------------------------------------------------
CAPE_PACKAGE_MAP: Dict[str, str] = {
    ".exe": "exe", ".dll": "dll", ".doc": "doc", ".docx": "doc",
    ".xls": "xls", ".xlsx": "xls", ".xlsm": "xls", ".pdf": "pdf",
    ".js": "js", ".vbs": "vbs", ".ps1": "ps1", ".bat": "exe",
    ".cmd": "exe", ".scr": "exe", ".msi": "exe", ".zip": "zip",
    ".rar": "zip", ".7z": "zip", ".elf": "elf", ".sh": "sh", ".py": "py",
}

PLATFORM_MAP: Dict[str, List[str]] = {
    ".exe": ["windows"], ".dll": ["windows"], ".doc": ["windows"],
    ".pdf": ["windows"], ".ps1": ["windows"], ".vbs": ["windows"],
    ".elf": ["linux"], ".sh": ["linux"], ".py": ["linux"],
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
    except:
        pass
    
    loc = response.headers.get("Location", "")
    m = re.search(r"/(\d+)/?$", loc)
    if m:
        return int(m.group(1))
    
    m = re.search(r'"task[_ ]?id"\s*:\s*(\d+)', response.text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def cape_is_alive() -> bool:
    try:
        r = requests.get(f"{CAPE_BASE}/apiv2/cuckoo/status/", headers=_headers(), timeout=REQUEST_TIMEOUT)
        return r.status_code == 200
    except:
        return False

# ---------------------------------------------------------------------------
# Submit file to CAPE
# ---------------------------------------------------------------------------
def submit_file(file_data: bytes, filename: str) -> Optional[int]:
    if not file_data or len(file_data) == 0:
        log.error("Empty file rejected")
        return None
    
    pkg = _resolve_package(filename) or "generic"
    ext = _get_extension(filename)
    plat = PLATFORM_MAP.get(ext, ["windows"])[0]
    
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"
    
    payload = {
        "package": pkg,
        "timeout": str(ANALYSIS_TIMEOUT),
        "options": "procmemdump=1,screenshots=1",
        "platforms": json.dumps([{"platform": plat, "os_version": ""}]),
    }
    
    files = {"file": (filename, file_data, mime)}
    
    for attempt in range(1, SUBMIT_RETRIES + 1):
        try:
            r = requests.post(CAPE_FILES, headers=_headers(), data=payload, files=files, timeout=REQUEST_TIMEOUT)
            if r.status_code in (200, 201):
                task_id = _extract_task_id(r)
                if task_id:
                    log.info(f"Submitted {filename} → Task ID: {task_id}")
                    return task_id
        except Exception as e:
            log.warning(f"Attempt {attempt} failed: {e}")
        
        if attempt < SUBMIT_RETRIES:
            time.sleep(2 ** attempt)
    
    return None

# ---------------------------------------------------------------------------
# Wait for task completion
# ---------------------------------------------------------------------------
def wait_for_task(task_id: int) -> Optional[str]:
    url = f"{CAPE_BASE}/apiv2/tasks/get/status/{task_id}/"
    terminal = {"reported", "failed_analysis", "failed_processing", "failed_reporting", "aborted"}
    
    for attempt in range(30):
        try:
            r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
            if r.ok:
                status = r.json().get("data", {}).get("status", "")
                log.info(f"Task {task_id} status: {status}")
                if status in terminal:
                    return status
        except:
            pass
        time.sleep(POLL_INTERVAL)
    
    return None

# ---------------------------------------------------------------------------
# Get report
# ---------------------------------------------------------------------------
def get_report(task_id: int) -> Optional[Dict]:
    url = CAPE_REPORT.format(task_id=task_id)
    try:
        r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
        return r.json() if r.ok else None
    except:
        return None

# ---------------------------------------------------------------------------
# Parse strace log
# ---------------------------------------------------------------------------
def parse_strace_log(analysis_id: int) -> Dict:
    strace_path = f"/home/rama/CAPEv2/storage/analyses/{analysis_id}/strace/strace.log"
    connections = []
    
    if os.path.exists(strace_path):
        with open(strace_path, 'r') as f:
            for line in f:
                if 'connect' in line and 'sin_port' in line:
                    ip_match = re.search(r'inet_addr\("([^"]+)"\)', line)
                    port_match = re.search(r'htons\((\d+)\)', line)
                    if ip_match and port_match:
                        connections.append({'ip': ip_match.group(1), 'port': int(port_match.group(1))})
    
    return {'has_connections': len(connections) > 0, 'connections': connections}

# ---------------------------------------------------------------------------
# Extract risk score
# ---------------------------------------------------------------------------
def extract_risk_score(report: Dict, task_id: int, task_status: Optional[str]) -> float:
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
# MAIN API - Called by mail_filter.py
# ---------------------------------------------------------------------------
def analyze_attachment(filename: str, file_data: bytes, message_id: str = None) -> Dict[str, Any]:
    """
    Main entry point for mail_filter.py - analyzes attachment with CAPE.
    """
    log.info(f"analyze_attachment: {filename} ({len(file_data)} bytes) msg={message_id}")
    
    # Handle empty files
    if not file_data or len(file_data) == 0:
        return {
            "filename": filename,
            "verdict": "skipped",
            "risk_score": 0.05,
            "payload_size": 0,
            "cape_verdicts": []
        }
    
    # Submit to CAPE
    task_id = submit_file(file_data, filename)
    if not task_id:
        return {
            "filename": filename,
            "verdict": "submission_failed",
            "risk_score": 0.40,
            "payload_size": len(file_data),
            "cape_verdicts": []
        }
    
    # Wait for analysis
    task_status = wait_for_task(task_id)
    
    # Get report
    report = get_report(task_id)
    
    # Extract risk score
    risk_score = extract_risk_score(report, task_id, task_status)
    
    # Parse strace if available
    strace_data = parse_strace_log(task_id) if task_id else {}
    
    # Build verdict
    verdict = "malicious" if risk_score >= 0.7 else "suspicious" if risk_score >= 0.4 else "clean"
    
    result = {
        "filename": filename,
        "verdict": verdict,
        "risk_score": risk_score,
        "payload_size": len(file_data),
        "task_id": task_id,
        "task_status": task_status,
        "cape_verdicts": [{
            "platform": "windows",
            "task_id": task_id,
            "malscore": report.get("malscore", 0) if report else 0,
            "risk_score": risk_score,
            "behaviors": strace_data.get("connections", []),
        }] if report else [],
    }
    
    log.info(f"Result for {filename}: verdict={verdict}, risk={risk_score:.3f}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sandbox_client.py <file_path>")
        sys.exit(1)
    
    with open(sys.argv[1], 'rb') as f:
        data = f.read()
    
    result = analyze_attachment(sys.argv[1], data)
    print(json.dumps(result, indent=2))
EOF

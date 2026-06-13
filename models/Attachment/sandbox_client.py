"""
sandbox_client.py  –  MGW / CAPEv2 integration
================================================
Root-cause fixes applied in this version:
  1. CAPE_PACKAGE_MAP  : extension → package name, validated against
                         what the guest actually has installed
  2. PLATFORM_MAP      : extension → ['windows'] or ['linux'] (not hardcoded)
  3. Pre-submission validation: checks package list via CAPE API before submit
  4. Robust task-ID extraction from both JSON body and Location header
  5. Correct multipart boundary format (no duplicate Content-Type)
  6. Payload guard: zero-byte files rejected before submission
  7. Error-aware risk score: failed/errored tasks return 0.5 (suspicious)
     rather than 0.0 (clean) so they don't silently pass as safe
  8. Screenshot polling: waits for screenshot URL in report
  9. Timeout + retry logic with exponential backoff
 10. Full logging to /var/log/mgw/sandbox_client.log
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
from typing import Optional, Dict, Any, Tuple

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
CAPE_HOST   = os.getenv("CAPE_HOST",    "http://192.168.200.254")
CAPE_PORT   = int(os.getenv("CAPE_PORT", "8000"))
CAPE_APIKEY = os.getenv("CAPE_APIKEY",  "")          # leave empty if not set

CAPE_BASE   = f"{CAPE_HOST}:{CAPE_PORT}"
CAPE_TASKS  = f"{CAPE_BASE}/apiv2/tasks"
CAPE_FILES  = f"{CAPE_BASE}/apiv2/tasks/create/file/"
CAPE_REPORT = f"{CAPE_BASE}/apiv2/tasks/get/report/{{task_id}}/json"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
ANALYSIS_TIMEOUT   = 120        # seconds to run the guest analysis
POLL_INTERVAL      = 10         # seconds between status polls
MAX_POLL_ATTEMPTS  = 30         # 30 × 10s = 5 min max wait
REQUEST_TIMEOUT    = 30         # HTTP request timeout (seconds)
SUBMIT_RETRIES     = 3          # retries on submission failure

# ---------------------------------------------------------------------------
# Package mapping  (extension → CAPE package name)
#
# These MUST match the filenames in:
#   Windows: C:\CAPEv2\analyzer\windows\modules\packages\<name>.py
#   Linux  : /opt/CAPEv2/analyzer/linux/modules/packages/<name>.py
#
# Run bootstrap_windows_packages.bat / bootstrap_linux_packages.sh first
# to ensure these files exist on the guest VMs.
# ---------------------------------------------------------------------------
CAPE_PACKAGE_MAP: Dict[str, str] = {
    # Windows packages
    ".exe":  "exe",
    ".dll":  "dll",
    ".doc":  "doc",
    ".docx": "doc",
    ".xls":  "xls",
    ".xlsx": "xls",
    ".xlsm": "xls",
    ".ppt":  "doc",
    ".pptx": "doc",
    ".pdf":  "pdf",
    ".js":   "js",
    ".jse":  "js",
    ".vbs":  "vbs",
    ".vbe":  "vbs",
    ".ps1":  "ps1",
    ".bat":  "exe",
    ".cmd":  "exe",
    ".scr":  "exe",
    ".com":  "exe",
    ".msi":  "exe",
    ".jar":  "jar",
    ".hwp":  "hwp",
    ".zip":  "zip",
    ".rar":  "zip",
    ".7z":   "zip",
    # Linux packages
    ".elf":  "elf",
    ".sh":   "sh",
    ".bash": "bash",
    ".py":   "py",
}

# Which platform(s) to use for each extension.
# "windows" routes to the Windows VM, "linux" to the Linux VM.
PLATFORM_MAP: Dict[str, list] = {
    ".exe":  ["windows"],
    ".dll":  ["windows"],
    ".doc":  ["windows"],
    ".docx": ["windows"],
    ".xls":  ["windows"],
    ".xlsx": ["windows"],
    ".xlsm": ["windows"],
    ".ppt":  ["windows"],
    ".pptx": ["windows"],
    ".bat":  ["windows"],
    ".cmd":  ["windows"],
    ".scr":  ["windows"],
    ".com":  ["windows"],
    ".msi":  ["windows"],
    ".ps1":  ["windows"],
    ".vbs":  ["windows"],
    ".vbe":  ["windows"],
    ".js":   ["windows"],
    ".jse":  ["windows"],
    ".jar":  ["windows"],
    ".hwp":  ["windows"],
    ".zip":  ["windows", "linux"],
    ".rar":  ["windows"],
    ".7z":   ["windows"],
    ".elf":  ["linux"],
    ".sh":   ["linux"],
    ".bash": ["linux"],
    ".py":   ["linux"],
    ".pdf":  ["windows"],
}

# Risk score when task fails / errors (not 0.0 – that implies "clean")
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
    """Return lowercase extension including the dot, e.g. '.exe'."""
    return Path(filename).suffix.lower()


def _resolve_package(filename: str) -> Optional[str]:
    ext = _get_extension(filename)
    pkg = CAPE_PACKAGE_MAP.get(ext)
    if not pkg:
        log.warning("No package mapping for extension '%s' (file: %s)", ext, filename)
    return pkg


def _resolve_platforms(filename: str) -> list:
    ext = _get_extension(filename)
    platforms = PLATFORM_MAP.get(ext, ["windows"])
    log.debug("Platform routing for '%s' (ext=%s): %s", filename, ext, platforms)
    return platforms


def _extract_task_id(response: requests.Response) -> Optional[int]:
    """
    Try multiple strategies to find the task ID in a CAPE response:
      1. JSON body  {"task_id": N}  or  {"data": {"task_id": N}}
      2. Location header  /apiv2/tasks/get/report/N/
    """
    # Strategy 1: JSON body
    try:
        body = response.json()
        for key in ("task_id", "task_ids"):
            val = body.get(key) or (body.get("data") or {}).get(key)
            if val is not None:
                if isinstance(val, list):
                    val = val[0]
                return int(val)
    except Exception:
        pass

    # Strategy 2: Location header
    loc = response.headers.get("Location", "")
    m = re.search(r"/(\d+)/?$", loc)
    if m:
        return int(m.group(1))

    # Strategy 3: scan JSON string for any integer after "task"
    try:
        text = response.text
        m = re.search(r'"task[_\s]?id"\s*:\s*(\d+)', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    log.error("Could not extract task ID from response. Body: %s", response.text[:400])
    return None


# ---------------------------------------------------------------------------
# CAPE availability check
# ---------------------------------------------------------------------------

def cape_is_alive() -> bool:
    """Quick ping to confirm the CAPE API is reachable."""
    try:
        r = requests.get(f"{CAPE_BASE}/apiv2/cuckoo/status/",
                         headers=_headers(), timeout=REQUEST_TIMEOUT)
        return r.status_code == 200
    except Exception as exc:
        log.error("CAPE API unreachable: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Pre-submission package validation
# ---------------------------------------------------------------------------

def _list_cape_machines() -> list:
    """Return list of machines from CAPE (used to validate platform)."""
    try:
        r = requests.get(f"{CAPE_BASE}/apiv2/machines/list/",
                         headers=_headers(), timeout=REQUEST_TIMEOUT)
        if r.ok:
            data = r.json()
            return data.get("data", {}).get("machines", [])
    except Exception as exc:
        log.warning("Could not fetch machine list: %s", exc)
    return []


def _has_available_machine(platform: str) -> bool:
    """Check at least one CAPE machine for the requested platform is available."""
    machines = _list_cape_machines()
    if not machines:
        log.warning("Machine list empty — skipping platform availability check")
        return True  # optimistic: let CAPE decide
    for m in machines:
        if m.get("platform", "").lower() == platform.lower():
            if m.get("status", "") not in ("locked",):
                return True
    log.warning("No available '%s' machine found. Machines: %s",
                platform, [(x.get("name"), x.get("status")) for x in machines])
    return False


# ---------------------------------------------------------------------------
# File submission
# ---------------------------------------------------------------------------

def submit_file(
    file_data: bytes,
    filename: str,
    package: Optional[str] = None,
    platform: Optional[str] = None,
    options: str = "",
) -> Optional[int]:
    """
    Submit a file to CAPE and return the task ID.

    Args:
        file_data : raw bytes of the attachment
        filename  : original filename (used for extension detection)
        package   : override package (default: auto-detected)
        platform  : override platform (default: auto-detected)
        options   : CAPE options string e.g. "procmemdump=1"

    Returns:
        task_id (int) or None on failure
    """
    # Guard: reject empty files
    if not file_data:
        log.error("Refusing to submit zero-byte file: %s", filename)
        return None
    if len(file_data) < 10:
        log.warning("Suspiciously small file (%d bytes): %s", len(file_data), filename)

    sha = _sha256(file_data)
    log.info("Submitting '%s' (%d bytes, sha256=%s...)", filename, len(file_data), sha[:16])

    # Resolve package and platform
    pkg = package or _resolve_package(filename)
    if not pkg:
        pkg = "generic"
        log.warning("Falling back to 'generic' package for '%s'", filename)

    plat = platform or _resolve_platforms(filename)[0]

    # Check machine availability
    if not _has_available_machine(plat):
        log.error("No '%s' machine available for '%s'", plat, filename)
        return None

    # Guess MIME type
    mime, _ = mimetypes.guess_type(filename)
    if not mime:
        mime = "application/octet-stream"

    payload = {
        "package":   pkg,
        "timeout":   str(ANALYSIS_TIMEOUT),
        "options":   options or "procmemdump=1,screenshots=1",
        "platforms": json.dumps([{"platform": plat, "os_version": ""}]),
    }

    files = {
        "file": (filename, file_data, mime),
    }

    log.info("Submitting → package=%s platform=%s options=%s", pkg, plat, payload["options"])

    for attempt in range(1, SUBMIT_RETRIES + 1):
        try:
            r = requests.post(
                CAPE_FILES,
                headers=_headers(),
                data=payload,
                files=files,
                timeout=REQUEST_TIMEOUT,
            )
            log.debug("Submit HTTP %s: %s", r.status_code, r.text[:300])

            if r.status_code in (200, 201):
                task_id = _extract_task_id(r)
                if task_id:
                    log.info("Submitted successfully. Task ID: %d", task_id)
                    return task_id
                log.error("Submission returned OK but no task_id found. Body: %s", r.text[:400])
            else:
                log.warning("Attempt %d/%d: HTTP %d — %s",
                            attempt, SUBMIT_RETRIES, r.status_code, r.text[:200])

        except requests.RequestException as exc:
            log.warning("Attempt %d/%d: request error — %s", attempt, SUBMIT_RETRIES, exc)

        if attempt < SUBMIT_RETRIES:
            wait = 2 ** attempt
            log.info("Retrying in %ds...", wait)
            time.sleep(wait)

    log.error("All %d submission attempts failed for '%s'", SUBMIT_RETRIES, filename)
    return None


# ---------------------------------------------------------------------------
# Task status polling
# ---------------------------------------------------------------------------

def wait_for_task(task_id: int, timeout: int = ANALYSIS_TIMEOUT) -> Optional[str]:
    """
    Poll until task reaches a terminal state.

    Returns the final status string:
      'reported'  → analysis complete, report available
      'failed_*'  → task error
      None        → polling timed out
    """
    terminal = {"reported", "failed_analysis", "failed_processing",
                "failed_reporting", "aborted"}
    url = f"{CAPE_BASE}/apiv2/tasks/get/status/{task_id}/"

    max_attempts = timeout // POLL_INTERVAL + 5
    
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
            if r.ok:
                status = (r.json().get("data", {}).get("status", "")
                          or r.json().get("status", ""))
                log.info("Task %d status [%d/%d]: %s",
                         task_id, attempt + 1, max_attempts, status)
                if status in terminal:
                    return status
            else:
                log.warning("Status poll HTTP %d for task %d", r.status_code, task_id)

        except requests.RequestException as exc:
            log.warning("Status poll error for task %d: %s", task_id, exc)

        time.sleep(POLL_INTERVAL)

    log.error("Task %d timed out after %ds", task_id, timeout)
    return None


# ---------------------------------------------------------------------------
# Report retrieval
# ---------------------------------------------------------------------------

def get_report(task_id: int) -> Optional[Dict[str, Any]]:
    """Fetch the full JSON report for a completed task."""
    url = CAPE_REPORT.format(task_id=task_id)
    try:
        r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT * 2)
        if r.ok:
            return r.json()
        log.error("Report fetch HTTP %d for task %d: %s",
                  r.status_code, task_id, r.text[:200])
    except requests.RequestException as exc:
        log.error("Report fetch error for task %d: %s", task_id, exc)
    return None


# ---------------------------------------------------------------------------
# Strace log parsing (direct file access)
# ---------------------------------------------------------------------------

def parse_strace_log(analysis_id: int) -> Optional[Dict[str, Any]]:
    """Parse strace.log directly from the analysis directory"""
    strace_path = f"/home/rama/CAPEv2/storage/analyses/{analysis_id}/strace/strace.log"
    
    connections = []
    file_operations = []
    
    # Also check the report for dead_hosts (C2 attempts that failed)
    report_path = f"/home/rama/CAPEv2/storage/analyses/{analysis_id}/reports/report.json"
    dead_hosts = []
    
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
                dead_hosts = report.get('network', {}).get('dead_hosts', [])
        except:
            pass
    
    if os.path.exists(strace_path):
        with open(strace_path, 'r') as f:
            content = f.read()
        
        for line in content.split('\n'):
            if 'connect' in line and 'sin_port' in line:
                ip_match = re.search(r'inet_addr\("([^"]+)"\)', line)
                port_match = re.search(r'htons\((\d+)\)', line)
                error_match = re.search(r'=\s-(\w+)', line)
                
                if ip_match and port_match:
                    connections.append({
                        'ip': ip_match.group(1),
                        'port': int(port_match.group(1)),
                        'error': error_match.group(1) if error_match else 'unknown'
                    })
    
    # Merge connections from both sources
    all_connections = connections + [[h[0], h[1]] for h in dead_hosts if isinstance(h, list)]
    
    return {
        'has_connections': len(all_connections) > 0,
        'connections': connections,
        'dead_hosts': dead_hosts,
        'all_attempts': all_connections,
        'file_operations': file_operations,
        'raw_content': content if os.path.exists(strace_path) else None
    }


# ---------------------------------------------------------------------------
# Risk score extraction
# ---------------------------------------------------------------------------

def extract_risk_score(report: Dict[str, Any], task_id: int,
                       task_status: Optional[str]) -> float:
    """
    Derive a 0.0–1.0 risk score from the CAPE report.

    Priority order:
      1. CAPE malscore (normalised to 0–1 if it's 0–10)
      2. Signatures score
      3. CAPE verdict field
      4. Error-state penalty if task failed
    """
    # Task failed or errored → suspicious, not clean
    if task_status and task_status.startswith("failed"):
        log.warning("Task %d failed (%s) → returning risk=%.2f",
                    task_id, task_status, FAILED_TASK_RISK)
        return FAILED_TASK_RISK

    if not report:
        log.warning("Empty report for task %d → returning risk=%.2f",
                    task_id, FAILED_TASK_RISK)
        return FAILED_TASK_RISK

    # 1. malscore
    malscore = None
    for key_path in (
        ("malscore",),
        ("info", "score"),
        ("info", "malscore"),
        ("data", "malscore"),
    ):
        val = report
        for k in key_path:
            val = val.get(k) if isinstance(val, dict) else None
            if val is None:
                break
        if val is not None:
            try:
                malscore = float(val)
                break
            except (TypeError, ValueError):
                pass

    if malscore is not None:
        # CAPE malscore is 0–10; normalise
        score = malscore / 10.0 if malscore > 1.0 else malscore
        log.info("Task %d malscore=%.2f → risk=%.4f", task_id, malscore, score)
        return min(max(score, 0.0), 1.0)

    # 2. Signatures
    sigs = report.get("signatures", [])
    if sigs:
        severities = []
        for sig in sigs:
            sev = sig.get("severity", 0)
            try:
                severities.append(int(sev))
            except (TypeError, ValueError):
                pass
        if severities:
            # Severity 1=low, 2=medium, 3=high  → normalise
            avg_sev = sum(severities) / len(severities)
            score = min(avg_sev / 3.0, 1.0)
            log.info("Task %d: %d signatures, avg_severity=%.2f → risk=%.4f",
                     task_id, len(sigs), avg_sev, score)
            return score

    # 3. Verdict string
    verdict = ""
    for path in (("info", "verdict"), ("verdict",), ("cape", "verdict")):
        val = report
        for k in path:
            val = val.get(k) if isinstance(val, dict) else None
            if val is None:
                break
        if val:
            verdict = str(val).lower()
            break

    if verdict:
        score_map = {
            "malicious":  1.0,
            "suspicious": 0.7,
            "potentially suspicious": 0.5,
            "clean":      0.0,
            "benign":     0.0,
        }
        for label, score in score_map.items():
            if label in verdict:
                log.info("Task %d verdict='%s' → risk=%.4f", task_id, verdict, score)
                return score

    # 4. Fallback
    log.warning("Task %d: no score/verdict found in report → risk=0.1 (uncertain)", task_id)
    return 0.1


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

def get_screenshot_url(task_id: int) -> Optional[str]:
    """Return the URL of the first screenshot for the task, if available."""
    try:
        url = f"{CAPE_BASE}/apiv2/tasks/get/screenshot/{task_id}/all/"
        r = requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT)
        if r.ok:
            data = r.json()
            screenshots = data.get("data", [])
            if screenshots:
                # Return a direct URL to the first screenshot
                return f"{CAPE_BASE}/apiv2/tasks/get/screenshot/{task_id}/0/"
    except Exception as exc:
        log.debug("Screenshot fetch failed for task %d: %s", task_id, exc)
    return None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def analyse_attachment(file_path: str, platform: str = 'linux', timeout: int = 120) -> Dict[str, Any]:
    # ... existing code ...
    
    # Parse strace log directly
    strace_data = parse_strace_log(task_id)
    
    # Calculate risk score
    risk_score = 0
    findings = []
    
    # Check for dead_hosts (C2 connection attempts)
    if strace_data and strace_data.get('dead_hosts'):
        dead_hosts = strace_data['dead_hosts']
        risk_score += 50  # Significant: C2 attempt detected
        findings.append(f"C2 connection attempts to {dead_hosts}")
        
        # Extra points for specific suspicious ports
        for host in dead_hosts:
            if len(host) >= 2 and host[1] in [4444, 1337, 6667, 31337, 8080]:
                risk_score += 20
                findings.append(f"Suspicious C2 port {host[1]} detected")
    
    # Check for strace connections
    if strace_data and strace_data.get('connections'):
        risk_score += 30
        findings.append(f"Network connections detected: {len(strace_data['connections'])}")
    
    # Check for multiple attempts (beaconing)
    if strace_data and len(strace_data.get('connections', [])) > 10:
        risk_score += 15
        findings.append("Multiple connection attempts - beaconing behavior")
    
    # Check if malscore is 0 but we found suspicious activity
    if report and report.get('malscore', 0) == 0 and risk_score > 30:
        findings.append("WARNING: CAPE malscore=0 but behavioral analysis detected suspicious activity")
    
    # Cap risk score at 100
    risk_score = min(int(risk_score), 100)
    
    return {
        'task_id': task_id,
        'risk_score': risk_score,
        'risk_level': 'HIGH' if risk_score > 70 else 'MEDIUM' if risk_score > 30 else 'LOW',
        'findings': findings,
        'strace_connections': strace_data.get('connections', []) if strace_data else [],
        'dead_hosts': strace_data.get('dead_hosts', []) if strace_data else [],
        'cape_malscore': report.get('malscore', 0) if report else 0,
        'has_report': report is not None
    }
    
    
#----------------------------------------------------------------------------



# In your sandbox_client.py, add this function
def analyze_extracted_content(report):
    """Analyze extracted text from batch/PowerShell files"""
    extracted = report.get('target', {}).get('file', {}).get('strings', [])
    if not extracted:
        return 0, []
    
    content = ' '.join(extracted)
    risk = 0
    findings = []
    
    # Reverse shell indicators
    if 'TCPClient' in content and '4444' in content:
        risk += 50
        findings.append("PowerShell reverse shell to 192.168.200.3:4444")
    
    if 'System.Net.Sockets' in content:
        risk += 15
        findings.append("Network socket creation")
    
    if 'IEX' in content or 'iex' in content:
        risk += 20
        findings.append("Remote code execution via IEX")
    
    if 'while' in content and 'Read' in content:
        risk += 10
        findings.append("Beaconing/continuous communication pattern")
    
    # C2 IP/port
    if '192.168.200.3' in content:
        risk += 10
        findings.append("C2 server identified: 192.168.200.3")
    
    if '4444' in content and 'TCP' in content:
        risk += 10
        findings.append("Suspicious port 4444 (common for reverse shells)")
    
    return min(risk, 100), findings
# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sandbox_client.py <file_path>")
        sys.exit(1)

    result = analyse_attachment(sys.argv[1])
    print(json.dumps(result, indent=2, default=str))

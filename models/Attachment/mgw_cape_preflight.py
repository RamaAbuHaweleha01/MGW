#!/usr/bin/env python3
"""
mgw_cape_preflight.py
=====================
Run this on the MGW host (192.168.200.254) to diagnose all known
CAPEv2 sandbox issues before deploying the fix.

Usage:
    python3 mgw_cape_preflight.py

It checks:
  1. CAPE API reachable
  2. Windows VM machine registered + status
  3. Linux VM machine registered + status
  4. Required packages exist on each guest (via CAPE machine details)
  5. sandbox_client.py config sanity
  6. Sample submission dry-run (EICAR test, no real malware)
"""

import os
import sys
import json
import time
import hashlib
import requests

CAPE_HOST = os.getenv("CAPE_HOST", "http://192.168.200.254")
CAPE_PORT = int(os.getenv("CAPE_PORT", "8000"))
CAPE_BASE = f"{CAPE_HOST}:{CAPE_PORT}"

REQUIRED_WIN_PACKAGES = [
    "exe", "dll", "doc", "xls", "pdf", "js", "vbs", "ps1", "zip", "jar"
]
REQUIRED_LIN_PACKAGES = [
    "elf", "sh", "bash", "py", "generic"
]

# EICAR test string (safe, universally detected as test malware)
EICAR = (
    b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
    b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
)

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
WARN = "\033[33m[WARN]\033[0m"
INFO = "\033[36m[INFO]\033[0m"


def req(path, **kwargs):
    try:
        r = requests.get(f"{CAPE_BASE}{path}", timeout=10, **kwargs)
        return r
    except Exception as e:
        return None


def check_api():
    print(f"\n{INFO} --- 1. CAPE API connectivity ---")
    r = req("/apiv2/cuckoo/status/")
    if r and r.status_code == 200:
        print(f"{PASS} API reachable at {CAPE_BASE}")
        try:
            d = r.json()
            version = d.get("data", {}).get("version", "?")
            print(f"       CAPE version: {version}")
        except Exception:
            pass
        return True
    else:
        code = r.status_code if r else "no response"
        print(f"{FAIL} API not reachable ({code}). Check:")
        print(f"       - Is CAPEv2 running? (systemctl status cape)")
        print(f"       - Is {CAPE_BASE} correct? (set CAPE_HOST env var)")
        return False


def check_machines():
    print(f"\n{INFO} --- 2. Virtual machine inventory ---")
    r = req("/apiv2/machines/list/")
    if not r or not r.ok:
        print(f"{FAIL} Could not fetch machine list")
        return [], []

    machines = r.json().get("data", {}).get("machines", [])
    if not machines:
        print(f"{FAIL} No machines registered with CAPE!")
        print(f"       Add VMs via: cuckoo machine --add ...")
        return [], []

    win_machines = []
    lin_machines = []
    for m in machines:
        name     = m.get("name", "?")
        platform = m.get("platform", "?")
        status   = m.get("status",   "?")
        snap     = m.get("snapshot", "?")
        icon = PASS if status not in ("locked", "error") else WARN
        print(f"  {icon} {name:20s} platform={platform:8s} status={status:10s} snapshot={snap}")
        if platform == "windows":
            win_machines.append(m)
        elif platform == "linux":
            lin_machines.append(m)

    if not win_machines:
        print(f"  {FAIL} No Windows machine found!")
    if not lin_machines:
        print(f"  {WARN} No Linux machine found (Linux analysis disabled)")

    return win_machines, lin_machines


def check_packages(machines, platform, required_packages):
    print(f"\n{INFO} --- 3. Package files on {platform} guest ---")
    if not machines:
        print(f"  {WARN} No {platform} machines to check — skipping")
        return

    # We can't SSH in from here, but we can check via CAPE's machine detail API
    for m in machines:
        name = m.get("name", "?")
        r = req(f"/apiv2/machines/view/{name}/")
        if r and r.ok:
            detail = r.json().get("data", {}).get("machine", {})
            options = detail.get("options", [])
            print(f"  Machine '{name}': options={options}")
        else:
            print(f"  {WARN} Could not get detail for machine '{name}'")

    print(f"\n  Required packages for {platform}:")
    print(f"  {required_packages}")
    print(f"  {WARN} Cannot verify packages remotely. Run the bootstrap script")
    print(f"       INSIDE the VM:")
    if platform == "windows":
        print(f"       bootstrap_windows_packages.bat")
    else:
        print(f"       sudo bash bootstrap_linux_packages.sh")


def check_sandbox_client():
    print(f"\n{INFO} --- 4. sandbox_client.py config ---")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sandbox_client", "./sandbox_client.py"
        )
        if spec is None:
            raise ImportError("sandbox_client.py not found in current directory")
        mod = importlib.util.load_module_from_spec(spec)
        spec.loader.exec_module(mod)

        host = getattr(mod, "CAPE_HOST", "?")
        port = getattr(mod, "CAPE_PORT", "?")
        pkg_map = getattr(mod, "CAPE_PACKAGE_MAP", {})
        plat_map = getattr(mod, "PLATFORM_MAP", {})

        print(f"  {PASS} sandbox_client.py loaded")
        print(f"       CAPE_HOST = {host}:{port}")
        print(f"       Package map: {len(pkg_map)} entries")
        print(f"       Platform map: {len(plat_map)} entries")

        # Check .exe is routed to windows
        if plat_map.get(".exe") == ["windows"]:
            print(f"  {PASS} .exe → windows routing correct")
        else:
            print(f"  {FAIL} .exe platform routing wrong: {plat_map.get('.exe')}")

        if plat_map.get(".elf") == ["linux"]:
            print(f"  {PASS} .elf → linux routing correct")
        else:
            print(f"  {WARN} .elf platform routing: {plat_map.get('.elf')} (expected ['linux'])")

    except Exception as e:
        print(f"  {FAIL} Could not load sandbox_client.py: {e}")
        print(f"       Make sure sandbox_client.py is in the current directory")


def dry_run_submission(win_machines, lin_machines):
    print(f"\n{INFO} --- 5. Dry-run submission (EICAR test file) ---")

    if not win_machines and not lin_machines:
        print(f"  {WARN} No machines available — skipping dry run")
        return

    # Pick a platform to test
    if win_machines:
        platform = "windows"
        pkg      = "exe"
        fname    = "eicar_test.exe"
    else:
        platform = "linux"
        pkg      = "elf"
        fname    = "eicar_test.elf"

    print(f"  Submitting EICAR as {fname} to {platform}...")
    try:
        url = f"{CAPE_BASE}/apiv2/tasks/create/file/"
        payload = {
            "package":   pkg,
            "timeout":   "60",
            "options":   "screenshots=1",
            "platforms": json.dumps([{"platform": platform, "os_version": ""}]),
        }
        files = {"file": (fname, EICAR, "application/octet-stream")}
        r = requests.post(url, data=payload, files=files, timeout=15)

        if r.status_code in (200, 201):
            try:
                body = r.json()
                task_id = body.get("task_id") or \
                          (body.get("data") or {}).get("task_id")
                if task_id:
                    print(f"  {PASS} Submitted! Task ID: {task_id}")
                    print(f"       Monitor: {CAPE_BASE}/analysis/{task_id}/")
                    return
            except Exception:
                pass
            print(f"  {WARN} Submitted (HTTP {r.status_code}) but no task_id in response:")
            print(f"       {r.text[:300]}")
        else:
            print(f"  {FAIL} Submission failed: HTTP {r.status_code}")
            print(f"       {r.text[:300]}")
            print(f"\n  Common causes:")
            print(f"    - Package '{pkg}' not found on guest VM")
            print(f"      → Run bootstrap script inside the VM first")
            print(f"    - No snapshot named 'cape_ready' on the VM")
            print(f"      → VBoxManage snapshot 'WinVM' take 'cape_ready' --pause")
            print(f"    - VM not restored to snapshot before analysis")
            print(f"      → Check cuckoo.conf: [cuckoo] → machinery = virtualbox")
            print(f"         and virtualbox.conf snapshot = cape_ready")

    except requests.RequestException as e:
        print(f"  {FAIL} Request error: {e}")


def main():
    print("=" * 60)
    print("  MGW / CAPEv2 Pre-flight Diagnostic")
    print(f"  Target: {CAPE_BASE}")
    print("=" * 60)

    if not check_api():
        print(f"\n{FAIL} CAPE API unreachable — cannot continue.")
        print("     Fix the API connection before proceeding.\n")
        sys.exit(1)

    win, lin = check_machines()
    check_packages(win, "windows", REQUIRED_WIN_PACKAGES)
    check_packages(lin, "linux",   REQUIRED_LIN_PACKAGES)
    check_sandbox_client()
    dry_run_submission(win, lin)

    print("\n" + "=" * 60)
    print("  Diagnostic complete. Review any [FAIL] or [WARN] lines.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

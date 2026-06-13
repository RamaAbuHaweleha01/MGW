#!/usr/bin/env python3
"""
Fix CAPE sandbox package issues for all file types.
Run this on MGW to patch both sandbox VMs.
"""

import os, json, subprocess, sys

CAPE_URL = "http://10.0.0.10:8000"

# Complete package mapping for all file types
PACKAGE_MAP = {
    # Windows packages
    ".exe":  ("CAPE_WIN",   "exe"),
    ".dll":  ("CAPE_WIN",   "dll"),
    ".ps1":  ("CAPE_WIN",   "ps1"),
    ".bat":  ("CAPE_WIN",   "bat"),
    ".cmd":  ("CAPE_WIN",   "bat"),
    ".doc":  ("CAPE_WIN",   "doc"),
    ".docx": ("CAPE_WIN",   "doc"),
    ".xls":  ("CAPE_WIN",   "xls"),
    ".xlsx": ("CAPE_WIN",   "xls"),
    ".pdf":  ("CAPE_WIN",   "pdf"),
    ".js":   ("CAPE_WIN",   "js"),
    ".vbs":  ("CAPE_WIN",   "vbs"),
    ".hta":  ("CAPE_WIN",   "hta"),
    # Linux packages
    ".elf":  ("CAPE_Linux", "generic"),
    ".sh":   ("CAPE_Linux", "generic"),
    # Generic fallback (decoded payloads)
    ".txt":  ("CAPE_WIN",   "generic"),
    ".hex":  ("CAPE_WIN",   "generic"),
    ".b64":  ("CAPE_WIN",   "generic"),
}

def get_package(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return PACKAGE_MAP.get(ext, ("CAPE_WIN", "generic"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_sandbox_packages.py <filepath>")
        sys.exit(1)
    fp = sys.argv[1]
    machine, pkg = get_package(fp)
    print(f"File    : {fp}")
    print(f"Machine : {machine}")
    print(f"Package : {pkg}")

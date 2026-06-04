#!/bin/bash

# ── deploy_instructions.sh ──────────────────────────────────────────────────

# Run this on MGW to deploy both updated files and fix all three issues.

# ─────────────────────────────────────────────────────────────────────────────

set -e
MGW="$HOME/MGW"

echo "[1/5] Creating correct directory layout..."
mkdir -p "$MGW/models/Attachment/sandbox"
mkdir -p "$MGW/models/Header"
mkdir -p "$MGW/models/Body"
mkdir -p "$MGW/models/Attachment"   # keep for backwards compat symlink


echo "[2/5] Backing up existing files..."
[ -f "$MGW/mail_filter.py" ]    && cp "$MGW/mail_filter.py"    "$MGW/mail_filter.py.bak"
[ -f "$MGW/models/Attachment/sandbox_client.py" ] && \
  cp "$MGW/models/Attachment/sandbox_client.py" \
     "$MGW/models/Attachment/sandbox_client.py.bak"
[ -f "$MGW/models/Attachment/sandbox_client.py" ] && \
  cp "$MGW/models/Attachment/sandbox_client.py" \
     "$MGW/models/Attachment/sandbox_client.py.bak"

echo "[3/5] Copying new files..."
# These paths assume you scp'd the files to /tmp/ first:

cp /tmp/mail_filter.py    "$MGW/mail_filter.py"
cp /tmp/sandbox_client.py "$MGW/models/Attachment/sandbox_client.py"

echo "[4/5] Verifying syntax..."
python3 -m py_compile "$MGW/mail_filter.py"    && echo "  mail_filter.py OK"
python3 -m py_compile "$MGW/models/Attachment/sandbox_client.py" && \
  echo "  sandbox_client.py OK"

echo "[5/5] Done. Restart mail_filter.py to pick up changes."
echo "  source ~/MGW/mail-env/bin/activate"
echo "  python ~/MGW/mail_filter.py"

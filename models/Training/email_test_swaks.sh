#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# test_swaks.sh — Send test emails to MGW on port 10025
# Run on MGW: bash test_swaks.sh
# ─────────────────────────────────────────────────────────────────
MGW="127.0.0.1"
PORT="10025"
FROM="attacker@evil.com"
TO="user@company.com"

# ── Install swaks if missing ──────────────────────────────────────
if ! command -v swaks &>/dev/null; then
    echo "[*] Installing swaks..."
    sudo apt-get install -y swaks 2>/dev/null || \
    sudo yum install -y swaks 2>/dev/null || \
    { echo "Install swaks manually: sudo apt install swaks"; exit 1; }
fi

# ── Test 1: Clean email (no attachment) ──────────────────────────
echo ""
echo "=== TEST 1: Clean email (no attachment) ==="
swaks \
  --server $MGW:$PORT \
  --from "$FROM" \
  --to "$TO" \
  --header "Subject: Monthly Newsletter" \
  --header "Authentication-Results: spf=pass dkim=pass dmarc=pass" \
  --body "Hello, please find our monthly newsletter attached. Best regards."
echo ""

# ── Test 2: Phishing email (no attachment) ───────────────────────
echo "=== TEST 2: Phishing email (no attachment) ==="
swaks \
  --server $MGW:$PORT \
  --from "support@paypa1-secure.tk" \
  --to "$TO" \
  --header "Subject: URGENT: Verify your account NOW!" \
  --header "Authentication-Results: spf=fail dkim=fail dmarc=fail" \
  --header "Reply-To: attacker@evil.xyz" \
  --body "<html><body>
  <p>URGENT ACTION REQUIRED: Your account has been suspended!</p>
  <p>Click <a href='http://1.2.3.4/login'>here</a> to verify immediately.</p>
  <p>Failure to verify within 24 hours will result in account termination.</p>
  <script>window.location='http://evil.tk/steal'</script>
  </body></html>"
echo ""

# ── Test 3: Email with Linux shell script attachment ─────────────
echo "=== TEST 3: Email with .sh attachment (→ CAPE_Linux) ==="
cat > /tmp/test_attach.sh << 'SHEOF'
#!/bin/bash
echo "CAPE Linux Analysis Test"
whoami
uname -a
sleep 5
ls /tmp
SHEOF

swaks \
  --server $MGW:$PORT \
  --from "$FROM" \
  --to "$TO" \
  --header "Subject: Invoice attached" \
  --header "Authentication-Results: spf=fail dkim=fail" \
  --body "Please find the invoice attached." \
  --attach-type "application/octet-stream" \
  --attach-name "invoice.sh" \
  --attach /tmp/test_attach.sh
echo ""

# ── Test 4: Email with Windows batch attachment ───────────────────
echo "=== TEST 4: Email with .bat attachment (→ CAPE_WIN) ==="
printf '@echo off\r\necho CAPE Windows Test\r\nwhoami\r\nipconfig\r\nping -n 3 127.0.0.1\r\n' > /tmp/test_attach.bat

swaks \
  --server $MGW:$PORT \
  --from "$FROM" \
  --to "$TO" \
  --header "Subject: Payment confirmation" \
  --header "Authentication-Results: spf=fail" \
  --body "Please run the attached payment tool." \
  --attach-type "application/octet-stream" \
  --attach-name "payment_tool.bat" \
  --attach /tmp/test_attach.bat
echo ""

# ── Test 5: Email with PDF attachment (→ both sandboxes) ─────────
echo "=== TEST 5: Email with .pdf attachment (→ both sandboxes) ==="
# Create a minimal valid-looking PDF stub
printf '%%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 2\n0000000000 65535 f \n%%EOF\n' > /tmp/test_attach.pdf

swaks \
  --server $MGW:$PORT \
  --from "$FROM" \
  --to "$TO" \
  --header "Subject: Your document is ready" \
  --body "Please review the attached document." \
  --attach-type "application/pdf" \
  --attach-name "document.pdf" \
  --attach /tmp/test_attach.pdf
echo ""

echo "=== All tests sent. Monitor logs with: ==="
echo "  tail -f ~/MGW/mail_filter.log"
echo "  tail -f ~/MGW/sandbox.log"
echo "  ls -lht ~/MGW/Attachment/ | head -20"

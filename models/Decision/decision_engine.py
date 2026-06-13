#!/usr/bin/env python3
"""
~/MGW/models/Decision/decision_engine.py
═══════════════════════════════════════════════════════════════════════════════
Decision Engine — MGW Phishing Defence Gateway

Scale contract (ALL internal values 0.0–1.0):
  - model risk_probability  → 0.0–1.0   (from header.py, body.py)
  - sandbox risk_score      → 0.0–1.0   (from sandbox_client.py)
  - CAPE malscore           → 0.0–10.0  → normalised ÷ 10 → 0.0–1.0
  - all weighted signals    → 0.0–1.0
  - composite_score_01      → 0.0–1.0   (internal arithmetic)
  - composite_score_10      → 0.0–10.0  (× 10, display only)

Actions: DROP | DELIVER  (fail-closed — no PEND)

Decision flow:
  1. Extract & normalise all signals → 0-1
  2. Weighted composite score        → 0-1
  3. Deterministic hard rules        → may force DROP
  4. Contradiction detection         → severe contradiction → DROP
  5. Final: DROP if composite ≥ T_DROP or rule/contradiction forced; else DELIVER

Sandbox veto:
  If any attachment has risk_score ≥ SANDBOX_VETO_THRESHOLD, the email is
  immediately DROPped regardless of the composite score or other models.
  This implements the "high sandbox risk → always drop" requirement.

Missing models default to 0.5 (suspicious-neutral) — never crashes on None.

FIXES applied vs previous version
───────────────────────────────────
FIX-DE01: T_CAPE_DROP split into two named thresholds (T_CAPE_HARD=0.70,
          T_CAPE_SOFT=0.40) — R-D01 uses HARD, R-D13 uses SOFT, no overlap.
FIX-DE02: R-D13 removed (was identical range to R-D01 with >=0.40) and
          replaced by T_CAPE_SOFT rule with distinct label R-D13.
FIX-DE03: cape_verdicts "behaviors" now parsed as either list-of-strings
          OR list-of-dicts {ip,port} from strace — no more dict.lower() crash.
FIX-DE04: Sandbox veto block added: attachment risk_score ≥ 0.70 → instant DROP.
FIX-DE05: Null guards added at explanation step (header/body_result may be None).
FIX-DE06: _detect_contradiction uses attach_result["risk_probability"] directly
          (already aggregated by mail_filter) — no longer reads .get("risk_probability")
          off the raw attachment list.
FIX-DE07: behavior_risk in cape_verdicts defaults to 0.0 when key absent
          (sandbox_client does not populate it — guard prevents KeyError).
FIX-DE08: T_DROP/T_DELIVER raised from 0.45 → 0.55 — previous threshold caused
          clean internal mail to be dropped because baseline signal noise
          (no-DKIM + model defaults) alone pushed composite past 0.45.
FIX-DE09: Missing model risk_probability default changed from 0.5 → 0.0 in ALL
          six locations (_extract_signals, _evaluate_rules, _detect_contradiction,
          sandbox veto block, final explanation block). "No model output" means
          no evidence of risk, not suspicious-neutral. The 0.5 default was the
          primary cause of legitimate mail being dropped when models are untrained.
FIX-DE10: no_dkim_signature weight reduced from 0.40 → 0.15. DKIM absence is
          a weak signal — internal Postfix/Dovecot setups typically do not sign
          outgoing mail, causing every internal email to score this signal at 1.0.
"""
from __future__ import annotations
import json, logging, os
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("decision_engine")

# ═══════════════════════════════════════════════════════════════════════════
# Thresholds — all in 0.0–1.0 space
# ═══════════════════════════════════════════════════════════════════════════
T_DROP               = 0.55   # composite ≥ this → DROP  (≡ 5.5/10)
T_DELIVER            = 0.55   # composite < this → DELIVER

# CAPE malscore thresholds (CAPE 0-10 → ÷10 → 0-1)
T_CAPE_HARD          = 0.70   # confirmed malware (≡ 7.0/10)  → R-D01
T_CAPE_SOFT          = 0.40   # suspicious sandbox flag (≡ 4.0/10) → R-D13

# Sandbox veto: any single attachment above this → instant DROP
# regardless of composite score or other models.
SANDBOX_VETO_THRESHOLD = 0.70   # matches T_CAPE_HARD on 0-1 scale

# ═══════════════════════════════════════════════════════════════════════════
# Adaptive signal weights  (all signals 0.0–1.0, weights are multipliers)
# composite = weighted_sum / total_weight  → stays in 0.0–1.0
# ═══════════════════════════════════════════════════════════════════════════
WEIGHTS = {
    # ── Authentication failures (0-1 flags) ─────────────────────────────
    "spf_fail":                   1.50,
    "dkim_fail":                  1.20,
    "dmarc_fail":                 1.30,
    "spf_softfail":               0.60,

    # ── Header / sender signals (0-1 flags) ──────────────────────────────
    "domain_mismatch":            1.40,
    "suspicious_tld_sender":      1.10,
    "has_numeric_in_domain":      0.50,
    "date_is_future":             0.80,
    "missing_message_id":         0.60,
    "missing_date":               0.50,
    "no_dkim_signature":          0.15,
    "display_name_spoofing":      1.20,

    # ── URL signals (0-1, some ratios clamped to 1.0) ────────────────────
    "url_has_ip":                 1.60,
    "url_mismatch":               1.50,
    "url_encoded_excess":         0.80,
    "url_suspicious_tld":         1.00,
    "url_shortener":              0.70,
    "url_redirect_chain_risk":    1.20,

    # ── Content signals (0-1 flags or ratios) ────────────────────────────
    "has_script":                 1.40,
    "has_iframe":                 1.30,
    "has_eval":                   1.60,
    "has_unescape":               1.40,
    "has_data_uri":               1.20,
    "has_form_with_password":     1.70,
    "has_form":                   0.80,
    "obfuscated_chars":           0.90,
    "urgency_score":              0.80,
    "fear_score":                 0.90,
    "curiosity_score":            0.60,
    "total_phishing_keywords":    0.05,   # per-ratio (clamped 0-1)
    "unique_phishing_keywords":   0.15,   # per-ratio (clamped 0-1)
    "money_symbols":              0.40,
    "subject_all_caps":           0.50,
    "subject_exclamation":        0.30,

    # ── Attachment / sandbox signals (0-1) ───────────────────────────────
    "attachment_win_executable":  2.00,
    "attachment_script":          1.60,
    "attachment_office_macro":    1.40,
    "attachment_archive":         0.80,
    "attachment_double_ext":      1.80,
    "attachment_linux_exec":      1.50,
    "cape_malscore_norm":         2.50,   # CAPE 0-10 ÷ 10
    "cape_behavior_risk":         2.00,   # 0-1
    "cape_network_contact":       0.80,   # 0-1 ratio
    "cape_dropped_files":         1.00,   # 0-1 ratio
    "cape_process_injection":     2.50,   # 0-1 flag

    # ── Model outputs (0-1 risk_probability) ─────────────────────────────
    "header_model_risk":          1.50,
    "body_model_risk":            1.80,

    # ── Routing signals ───────────────────────────────────────────────────
    "excessive_received_hops":    0.40,
    "received_from_suspicious":   0.60,
}


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DecisionResult:
    action:              str                   # "DROP" | "DELIVER"
    composite_score_01:  float                 # 0.0–1.0
    composite_score_10:  float                 # 0.0–10.0 (display)
    confidence:          str                   # "HIGH" | "MEDIUM" | "LOW"
    triggered_rules:     list  = field(default_factory=list)
    score_breakdown:     dict  = field(default_factory=dict)
    contradiction:       dict  = field(default_factory=dict)
    explanation:         str   = ""
    sandbox_veto:        bool  = False         # True when attachment vetoed
    timestamp:           str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "action":             self.action,
            "composite_score":    round(self.composite_score_10, 3),
            "composite_score_01": round(self.composite_score_01, 4),
            "confidence":         self.confidence,
            "triggered_rules":    self.triggered_rules,
            "score_breakdown":    self.score_breakdown,
            "contradiction":      self.contradiction,
            "explanation":        self.explanation,
            "sandbox_veto":       self.sandbox_veto,
            "timestamp":          self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Sandbox veto check
# ═══════════════════════════════════════════════════════════════════════════
def _check_sandbox_veto(attach_result: dict) -> tuple[bool, list[str]]:
    """
    FIX-DE04: Scan every attachment's risk_score. If any is at or above
    SANDBOX_VETO_THRESHOLD, return (True, reasons). This fires before
    the composite score is even considered — instant DROP.
    """
    if not attach_result:
        return False, []

    veto_reasons = []
    for att in attach_result.get("attachments", []):
        score = float(att.get("risk_score", 0.0))
        fname = att.get("filename", "?")
        if score >= SANDBOX_VETO_THRESHOLD:
            veto_reasons.append(
                f"R-VETO: Attachment '{fname}' sandbox risk={score:.4f} "
                f"≥ {SANDBOX_VETO_THRESHOLD} — sandbox veto DROP"
            )
    return bool(veto_reasons), veto_reasons


# ═══════════════════════════════════════════════════════════════════════════
# Signal extractor — all outputs 0.0–1.0
# ═══════════════════════════════════════════════════════════════════════════
def _extract_signals(semantic_meta, header_result, body_result, attach_result) -> dict:
    """
    Flatten all model outputs + semantic metadata into a single signal dict.
    Every value MUST be in [0.0, 1.0].

    FIX-DE03: cape_verdicts "behaviors" handled as either:
      - list of str  → standard CAPE behavior labels
      - list of dict → strace {ip, port} records from sandbox_client.parse_strace_log
    FIX-DE07: behavior_risk key absence defaults to 0.0.
    """
    s = semantic_meta  or {}
    h = header_result  or {}
    b = body_result    or {}
    a = attach_result  or {}

    # Model outputs — default to 0.0 (no evidence) when absent
    h_risk = float(h.get("risk_probability", 0.0))
    b_risk = float(b.get("risk_probability", 0.0))

    cape_malscore = 0.0
    cape_behavior = 0.0
    cape_net      = 0.0
    cape_dropped  = 0.0
    cape_inject   = 0.0
    win_exec      = 0.0
    lin_exec      = 0.0
    script_att    = 0.0
    office_macro  = 0.0
    archive_att   = 0.0
    double_ext    = 0.0

    WIN_EXEC_EXT = {".exe",".msi",".bat",".cmd",".lnk",".dll"}
    SCRIPT_EXT   = {".ps1",".vbs",".js",".wsf",".hta"}
    MACRO_EXT    = {".docm",".xlsm",".pptm",".rtf"}
    ARCHIVE_EXT  = {".zip",".rar",".7z",".iso",".img"}
    LINUX_EXT    = {".sh",".elf",".bin",".run",".deb",".rpm"}

    for att in a.get("attachments", []):
        fname = (att.get("filename") or "").lower()
        ext   = os.path.splitext(fname)[1]
        base  = os.path.splitext(fname)[0]

        # Double extension: invoice.pdf.exe
        if os.path.splitext(base)[1] in {".pdf",".doc",".xls",".png",".jpg"}:
            double_ext = 1.0

        if ext in WIN_EXEC_EXT:  win_exec    = 1.0
        if ext in SCRIPT_EXT:   script_att   = 1.0
        if ext in MACRO_EXT:    office_macro = 1.0
        if ext in ARCHIVE_EXT:  archive_att  = max(archive_att, 0.5)
        if ext in LINUX_EXT:    lin_exec     = 1.0

        for cv in att.get("cape_verdicts", []):
            # Normalise CAPE malscore 0-10 → 0-1
            ms = float(cv.get("malscore", 0)) / 10.0
            cape_malscore = max(cape_malscore, min(ms, 1.0))

            # FIX-DE07: behavior_risk may be absent — default 0.0
            br = float(cv.get("behavior_risk", 0.0))
            cape_behavior = max(cape_behavior, min(br, 1.0))

            net_hosts   = len(cv.get("network", {}).get("hosts", []))
            cape_net    = min(1.0, net_hosts / 5.0)

            dropped     = int(cv.get("dropped_files", 0))
            cape_dropped = min(1.0, dropped / 3.0)

            # FIX-DE03: behaviors may be list[str] (CAPE) or list[dict] (strace)
            raw_behaviors = cv.get("behaviors", [])
            for beh in raw_behaviors:
                if isinstance(beh, str):
                    if "process_injection" in beh.lower():
                        cape_inject = 1.0
                elif isinstance(beh, dict):
                    # strace {ip, port} — network contact, not process injection
                    cape_net = min(1.0, cape_net + 0.2)

        # Redirect chain risk — already 0-1
        for rd in att.get("redirect_chain", []):
            cape_behavior = max(cape_behavior, min(float(rd.get("risk_score", 0)), 1.0))

    # URL signals — ratios clamped to 0-1
    url_ip       = float(s.get("url_has_ip", 0))
    url_mismatch = min(1.0, int(s.get("url_mismatch_count", 0)) / 3.0)
    url_enc      = min(1.0, int(s.get("url_encoded_count",  0)) / 5.0)
    url_tld      = min(1.0, int(s.get("url_suspicious_tlds",0)) / 3.0)

    # Keyword density — 0-1 ratios
    total_kw  = min(1.0, int(s.get("total_phishing_keywords",  0)) / 20.0)
    unique_kw = min(1.0, int(s.get("unique_phishing_keywords", 0)) / 10.0)
    money_sym = min(1.0, int(s.get("total_money_symbols",      0)) / 5.0)

    subj_excl   = min(1.0, int(s.get("subject_exclamation", 0)) / 3.0)
    hops        = int(s.get("received_hops", 0))
    excess_hops = 1.0 if hops > 10 else 0.0

    return {
        "spf_fail":                  float(s.get("spf_fail",  0)),
        "dkim_fail":                 float(s.get("dkim_fail", 0)),
        "dmarc_fail":                float(s.get("dmarc_fail",0)),
        "spf_softfail":              0.0,
        "domain_mismatch":           float(s.get("domain_mismatch",       0)),
        "suspicious_tld_sender":     float(s.get("suspicious_tld_sender", 0)),
        "has_numeric_in_domain":     float(s.get("has_numeric_in_domain", 0)),
        "date_is_future":            float(s.get("date_is_future",        0)),
        "missing_message_id":        float(1 - int(s.get("has_message_id", 1))),
        "missing_date":              float(1 - int(s.get("has_date",       1))),
        "no_dkim_signature":         float(1 - int(s.get("has_dkim",       1))),
        "display_name_spoofing":     float(s.get("domain_mismatch",        0)),
        "url_has_ip":                url_ip,
        "url_mismatch":              url_mismatch,
        "url_encoded_excess":        url_enc,
        "url_suspicious_tld":        url_tld,
        "url_shortener":             0.0,
        "url_redirect_chain_risk":   cape_behavior if float(a.get("risk_probability", 0)) > 0 else 0.0,
        "has_script":                float(s.get("has_script",         0)),
        "has_iframe":                float(s.get("has_iframe",         0)),
        "has_eval":                  float(s.get("has_eval",           0)),
        "has_unescape":              float(s.get("has_unescape",       0)),
        "has_data_uri":              float(s.get("has_data_uri",       0)),
        "has_form_with_password":    float(s.get("has_input_password", 0)),
        "has_form":                  float(s.get("has_form",           0)),
        "obfuscated_chars":          float(s.get("obfuscated_chars",   0)),
        "urgency_score":             min(1.0, float(s.get("urgency_score",   0))),
        "fear_score":                min(1.0, float(s.get("fear_score",      0))),
        "curiosity_score":           min(1.0, float(s.get("curiosity_score", 0))),
        "total_phishing_keywords":   total_kw,
        "unique_phishing_keywords":  unique_kw,
        "money_symbols":             money_sym,
        "subject_all_caps":          float(s.get("subject_all_caps", 0)),
        "subject_exclamation":       subj_excl,
        "attachment_win_executable": win_exec,
        "attachment_script":         script_att,
        "attachment_office_macro":   office_macro,
        "attachment_archive":        archive_att,
        "attachment_double_ext":     double_ext,
        "attachment_linux_exec":     lin_exec,
        "cape_malscore_norm":        cape_malscore,
        "cape_behavior_risk":        cape_behavior,
        "cape_network_contact":      min(1.0, cape_net),
        "cape_dropped_files":        cape_dropped,
        "cape_process_injection":    cape_inject,
        "header_model_risk":         h_risk,
        "body_model_risk":           b_risk,
        "excessive_received_hops":   excess_hops,
        "received_from_suspicious":  0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Weighted scoring — composite_01 = weighted_sum / total_weight → 0.0–1.0
# ═══════════════════════════════════════════════════════════════════════════
def _compute_score(signals: dict) -> tuple:
    total_weight = sum(WEIGHTS.values())
    weighted_sum = 0.0
    breakdown    = {}

    for signal, weight in WEIGHTS.items():
        value = float(signals.get(signal, 0.0))
        value = max(0.0, min(1.0, value))
        contribution = value * weight
        weighted_sum += contribution
        if contribution > 0.005:
            breakdown[signal] = {
                "signal_value": round(value, 4),
                "weight":       weight,
                "contribution": round(contribution, 4),
            }

    composite_01 = round(min(weighted_sum / total_weight, 1.0), 6)
    composite_10 = round(composite_01 * 10.0, 3)
    return composite_01, composite_10, breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic rule engine — all comparisons in 0.0–1.0 space
# ═══════════════════════════════════════════════════════════════════════════
def _evaluate_rules(signals, composite_01, semantic_meta,
                    attach_result, header_result, body_result) -> tuple:
    """
    FIX-DE01/DE02: T_CAPE_HARD (0.70) used for R-D01 (confirmed malware);
    T_CAPE_SOFT (0.40) used for R-D13 (suspicious flag).  No overlap — R-D13
    only fires for malscore in [0.40, 0.70); R-D01 fires for > 0.70.
    """
    drop_reasons = []
    _h = header_result or {}
    _b = body_result   or {}
    _sm = semantic_meta or {}

    h_risk  = float(_h.get("risk_probability", 0.0))
    b_risk  = float(_b.get("risk_probability", 0.0))
    cape_01 = float(signals.get("cape_malscore_norm", 0.0))

    # ── HARD DROP RULES ──────────────────────────────────────────────────────
    # R-D01: CAPE malscore > 0.70 (≡ 7.0/10) — confirmed malware
    if cape_01 > T_CAPE_HARD:
        drop_reasons.append(
            f"R-D01: CAPE malscore {cape_01*10:.1f}/10 > 7.0 — sandbox confirmed malware"
        )

    # R-D02: Process injection in sandbox
    if signals.get("cape_process_injection", 0) > 0:
        drop_reasons.append("R-D02: Process injection detected by sandbox")

    # R-D03: Executable + all three auth failures
    if (signals.get("attachment_win_executable", 0) > 0 and
            signals.get("spf_fail",  0) > 0 and
            signals.get("dkim_fail", 0) > 0 and
            signals.get("dmarc_fail",0) > 0):
        drop_reasons.append(
            "R-D03: Executable attachment + SPF+DKIM+DMARC all failed")

    # R-D04: Absolute model consensus (both > 0.90)
    if h_risk > 0.90 and b_risk > 0.90:
        drop_reasons.append(
            f"R-D04: Model consensus — header={h_risk:.3f} body={b_risk:.3f}")

    # R-D05: IP URL + credential form
    if signals.get("url_has_ip", 0) > 0 and signals.get("has_form_with_password", 0) > 0:
        drop_reasons.append(
            "R-D05: IP-based URL + password input form (credential harvesting)")

    # R-D06: Double extension executable
    if signals.get("attachment_double_ext", 0) > 0 and signals.get("attachment_win_executable", 0) > 0:
        drop_reasons.append(
            "R-D06: Double-extension executable (e.g. invoice.pdf.exe)")

    # R-D07: eval() + script tag
    if signals.get("has_eval", 0) > 0 and signals.get("has_script", 0) > 0:
        drop_reasons.append(
            "R-D07: JavaScript eval() + <script> — code execution in email")

    # R-D08: data URI + script
    if signals.get("has_data_uri", 0) > 0 and signals.get("has_script", 0) > 0:
        drop_reasons.append(
            "R-D08: data: URI + <script> — inline payload delivery")

    # R-D09: Domain mismatch + suspicious TLD + SPF fail
    if (signals.get("domain_mismatch",       0) > 0 and
            signals.get("suspicious_tld_sender", 0) > 0 and
            signals.get("spf_fail",             0) > 0):
        drop_reasons.append(
            "R-D09: Domain mismatch + suspicious TLD + SPF fail")

    # R-D10: CAPE dropped files + network contact
    if signals.get("cape_dropped_files", 0) > 0 and signals.get("cape_network_contact", 0) > 0:
        drop_reasons.append(
            "R-D10: Sandbox dropped files AND contacted external hosts")

    # R-D11: IP URL + body model > 0.85
    if signals.get("url_has_ip", 0) > 0 and b_risk > 0.85:
        drop_reasons.append(
            f"R-D11: IP-based URL + high body risk ({b_risk:.3f})")

    # R-D12: 3+ URL anchor mismatches
    if int(_sm.get("url_mismatch_count", 0)) >= 3:
        drop_reasons.append(
            f"R-D12: {_sm.get('url_mismatch_count')} URL anchor mismatches")

    # R-D13: CAPE malscore in [0.40, 0.70) — suspicious sandbox flag
    # FIX-DE02: distinct from R-D01 (which covers > 0.70)
    if T_CAPE_SOFT <= cape_01 <= T_CAPE_HARD:
        drop_reasons.append(
            f"R-D13: CAPE malscore {cape_01*10:.1f}/10 in [4.0,7.0] — sandbox flagged suspicious")

    # R-D14: Body high risk, header clean
    if b_risk > 0.70 and h_risk < 0.35:
        drop_reasons.append(
            f"R-D14: Body risk high ({b_risk:.3f}) but header clean ({h_risk:.3f})")

    # R-D15: Header high risk, body clean
    if h_risk > 0.70 and b_risk < 0.35:
        drop_reasons.append(
            f"R-D15: Header risk high ({h_risk:.3f}) but body clean ({b_risk:.3f})")

    # R-D16: Executable attachment + no email auth
    if (signals.get("attachment_win_executable", 0) > 0 and
            signals.get("spf_fail",  0) == 0 and
            signals.get("dkim_fail", 0) == 0 and
            int(_sm.get("has_dkim", 0)) == 0):
        drop_reasons.append(
            "R-D16: Executable attachment + no SPF/DKIM auth present")

    # R-D17: Office macro + any auth failure
    if signals.get("attachment_office_macro", 0) > 0 and (
            signals.get("spf_fail", 0) > 0 or signals.get("dkim_fail", 0) > 0):
        drop_reasons.append("R-D17: Office macro attachment + auth failure")

    # R-D18: Archive + high body risk
    if signals.get("attachment_archive", 0) > 0 and b_risk > 0.65:
        drop_reasons.append(
            f"R-D18: Archive attachment + high body risk ({b_risk:.3f})")

    # R-D19: Future date + domain mismatch
    if signals.get("date_is_future", 0) > 0 and signals.get("domain_mismatch", 0) > 0:
        drop_reasons.append("R-D19: Future timestamp + domain mismatch")

    # R-D20: <iframe> + <form> in email body
    if signals.get("has_iframe", 0) > 0 and signals.get("has_form", 0) > 0:
        drop_reasons.append("R-D20: <iframe> + <form> in email body")

    # R-D21: High urgency + fear + keyword density
    if (float(_sm.get("urgency_score",            0)) > 0.60 and
            float(_sm.get("fear_score",           0)) > 0.60 and
            int(_sm.get("unique_phishing_keywords",0)) >= 5):
        drop_reasons.append("R-D21: High urgency + fear + 5+ phishing keywords")

    # R-D22: IP URL + no Message-ID
    if signals.get("url_has_ip", 0) > 0 and int(_sm.get("has_message_id", 1)) == 0:
        drop_reasons.append("R-D22: IP-based URL + missing Message-ID")

    # R-D23: Sandbox attachment contacted external hosts
    if signals.get("cape_network_contact", 0) > 0:
        drop_reasons.append("R-D23: Sandbox attachment contacted external hosts")

    # R-D24: Script attachment (.js/.vbs/.ps1/.hta)
    if signals.get("attachment_script", 0) > 0:
        drop_reasons.append("R-D24: Script file attachment (.js/.vbs/.ps1/.hta)")

    # R-D25: Suspicious sender TLD + urgency language
    if (signals.get("suspicious_tld_sender", 0) > 0 and
            float(_sm.get("urgency_score", 0)) > 0.40):
        drop_reasons.append("R-D25: Suspicious sender TLD + urgency language")

    # R-D26: Excessive routing hops
    if signals.get("excessive_received_hops", 0) > 0:
        drop_reasons.append(
            f"R-D26: Excessive routing hops ({_sm.get('received_hops', 0)})")

    # R-D27: Linux sandbox malscore ≥ T_CAPE_SOFT
    _at = attach_result or {}
    for att in _at.get("attachments", []):
        for cv in att.get("cape_verdicts", []):
            if cv.get("platform") == "linux":
                cape_linux_01 = min(float(cv.get("malscore", 0)) / 10.0, 1.0)
                if cape_linux_01 >= T_CAPE_SOFT:
                    drop_reasons.append(
                        f"R-D27: Linux sandbox malscore {cape_linux_01*10:.1f}/10 ≥ 4.0")

    # R-D28: Linux executable attachment
    if signals.get("attachment_linux_exec", 0) > 0:
        drop_reasons.append("R-D28: Linux executable attachment (.elf/.sh/.bin)")

    if drop_reasons:
        return drop_reasons, "DROP"
    return [], None


# ═══════════════════════════════════════════════════════════════════════════
# Contradiction detector
# ═══════════════════════════════════════════════════════════════════════════
def _detect_contradiction(header_result, body_result, attach_result) -> dict:
    _h = header_result or {}
    _b = body_result   or {}
    _a = attach_result or {}
    h_risk  = float(_h.get("risk_probability", 0.0))
    b_risk  = float(_b.get("risk_probability", 0.0))
    # attach_result top-level risk_probability is aggregated by mail_filter
    a_risk  = float(_a.get("risk_probability", 0.0))
    has_att = _a.get("has_attachments", 0)

    contradictions = []
    gap = abs(h_risk - b_risk)

    if gap > 0.45:
        contradictions.append({
            "type":        "header_body_gap",
            "description": (
                f"Header ({h_risk:.3f}) and body ({b_risk:.3f}) diverge by "
                f"{gap:.3f} — possible social engineering"
            ),
            "severity": "HIGH" if gap > 0.55 else "MEDIUM",
        })

    if has_att and a_risk == 0.0 and (h_risk > 0.65 or b_risk > 0.65):
        contradictions.append({
            "type":        "clean_attachment_suspicious_content",
            "description": (
                "Attachment is clean but email content is suspicious — "
                "possible decoy or unanalysed payload type"
            ),
            "severity": "MEDIUM",
        })

    if has_att and a_risk > 0.70 and h_risk < 0.30 and b_risk < 0.30:
        contradictions.append({
            "type":        "clean_content_malicious_attachment",
            "description": (
                "Clean email content but high-risk attachment — "
                "targeted attack with benign message body"
            ),
            "severity": "HIGH",
        })

    return {
        "has_contradiction": len(contradictions) > 0,
        "count":             len(contradictions),
        "details":           contradictions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Confidence calculator
# ═══════════════════════════════════════════════════════════════════════════
def _calculate_confidence(composite_01, rules, contradiction, sandbox_veto) -> str:
    if sandbox_veto:
        return "HIGH"
    if contradiction["has_contradiction"] and contradiction["count"] >= 2:
        return "LOW"
    if len(rules) >= 2 and not contradiction["has_contradiction"]:
        return "HIGH"
    if (composite_01 >= T_DROP or composite_01 < 0.30) and len(rules) >= 1:
        return "HIGH"
    if abs(composite_01 - T_DROP) < 0.05:
        return "LOW"
    return "MEDIUM"


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def decide(semantic_meta, header_result, body_result, attach_result) -> DecisionResult:
    """
    Main entry point called by mail_filter.py.

    Actions: "DROP" or "DELIVER" only — fail-closed.

    Sandbox veto (FIX-DE04): if any attachment risk_score ≥ SANDBOX_VETO_THRESHOLD
    the email is DROPped immediately before composite scoring even runs.
    """
    # 0. Sandbox veto — checked first, highest priority
    sandbox_veto, veto_reasons = _check_sandbox_veto(attach_result)
    if sandbox_veto:
        # Still compute score for logging, but action is forced
        signals = _extract_signals(semantic_meta, header_result, body_result, attach_result)
        composite_01, composite_10, breakdown = _compute_score(signals)
        contradiction = _detect_contradiction(header_result, body_result, attach_result)
        # FIX-DE05: null-guard at explanation step
        _h = header_result or {}
        _b = body_result   or {}
        _a = attach_result or {}
        h_risk = float(_h.get("risk_probability", 0.0))
        b_risk = float(_b.get("risk_probability", 0.0))
        a_risk = float(_a.get("risk_probability", 0.0))
        explanation = (
            f"SANDBOX VETO: attachment risk ≥ {SANDBOX_VETO_THRESHOLD}  |  "
            f"composite={composite_01:.4f} ({composite_10:.2f}/10)  |  "
            f"header={h_risk:.4f}  body={b_risk:.4f}  attach={a_risk:.4f}  |  "
            f"action=DROP  confidence=HIGH"
        )
        return DecisionResult(
            action             = "DROP",
            composite_score_01 = composite_01,
            composite_score_10 = composite_10,
            confidence         = "HIGH",
            triggered_rules    = veto_reasons,
            score_breakdown    = breakdown,
            contradiction      = contradiction,
            explanation        = explanation,
            sandbox_veto       = True,
        )

    # 1. Extract normalised signals
    signals = _extract_signals(semantic_meta, header_result, body_result, attach_result)

    # 2. Weighted composite
    composite_01, composite_10, breakdown = _compute_score(signals)

    # 3. Deterministic rules
    triggered_rules, forced_action = _evaluate_rules(
        signals, composite_01, semantic_meta,
        attach_result, header_result, body_result,
    )

    # 4. Contradiction detection
    contradiction = _detect_contradiction(header_result, body_result, attach_result)

    # 5. Final arbiter — DROP or DELIVER only
    if forced_action == "DROP" or composite_01 >= T_DROP:
        action = "DROP"
    else:
        if (contradiction["has_contradiction"] and
                any(c["severity"] == "HIGH" for c in contradiction["details"])):
            action = "DROP"
            triggered_rules.append(
                "R-C01: High-severity model contradiction → DROP (fail-closed)"
            )
        else:
            action = "DELIVER"

    # 6. Confidence
    confidence = _calculate_confidence(composite_01, triggered_rules, contradiction, False)

    # 7. Explanation — FIX-DE05: null-guard on model results
    _h = header_result or {}
    _b = body_result   or {}
    _a = attach_result or {}
    h_risk = float(_h.get("risk_probability", 0.0))
    b_risk = float(_b.get("risk_probability", 0.0))
    a_risk = float(_a.get("risk_probability", 0.0))

    explanation = (
        f"Composite: {composite_01:.4f} (0-1) = {composite_10:.2f}/10  |  "
        f"header={h_risk:.4f}  body={b_risk:.4f}  attach={a_risk:.4f}  |  "
        f"rules={len(triggered_rules)}  |  "
        f"contradiction={contradiction['count']}  |  "
        f"action={action}  confidence={confidence}"
    )

    return DecisionResult(
        action             = action,
        composite_score_01 = composite_01,
        composite_score_10 = composite_10,
        confidence         = confidence,
        triggered_rules    = triggered_rules,
        score_breakdown    = breakdown,
        contradiction      = contradiction,
        explanation        = explanation,
        sandbox_veto       = False,
    )

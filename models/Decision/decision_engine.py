#!/usr/bin/env python3
"""
~/MGW/models/Decision/decision_engine.py
═══════════════════════════════════════════════════════════════════════════════
Comprehensive Decision Engine — MGW Phishing Defence Gateway

Scale contract (ALL internal values 0.0 – 1.0):
  - model risk_probability  → 0.0–1.0   (from header.py, body.py)
  - sandbox risk_score      → 0.0–1.0   (from sandbox_client.py)
  - CAPE malscore           → 0.0–10.0  → normalised ÷ 10 → 0.0–1.0
  - all weighted signals    → 0.0–1.0
  - composite_score_01      → 0.0–1.0   (internal arithmetic)
  - composite_score_10      → 0.0–10.0  (× 10, display only)

Decision thresholds (0.0–1.0 space):
  DROP    : composite_01 ≥ 0.75   OR  any hard-DROP rule fires
  PEND    : composite_01 0.45–0.74  OR  any PEND rule fires
  DELIVER : composite_01 < 0.45   AND  no rules fired
"""
from __future__ import annotations
import json, logging, os
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("decision_engine")

# ═══════════════════════════════════════════════════════════════════════════
# ALL thresholds in 0.0–1.0 space
# ═══════════════════════════════════════════════════════════════════════════
T_DROP         = 0.75    # composite ≥ this → DROP    (≡ 7.5/10)
T_PEND_HI      = 0.749   # composite ≤ this (upper PEND boundary)
T_PEND_LO      = 0.45    # composite ≥ this → PEND    (≡ 4.5/10)
T_DELIVER      = 0.45    # composite < this → DELIVER (≡ 4.5/10)

# CAPE malscore thresholds — CAPE returns 0–10, we normalise ÷ 10
# so 7.0 → 0.70, 4.0 → 0.40
T_CAPE_DROP    = 0.70    # normalised CAPE malscore  (≡ 7.0/10)
T_CAPE_PEND    = 0.40    # normalised CAPE malscore  (≡ 4.0/10)

# ═══════════════════════════════════════════════════════════════════════════
# Adaptive signal weights  (all signals are 0.0–1.0, weights are multipliers)
# Composite = weighted_sum / total_weight  → stays in 0.0–1.0
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
    "no_dkim_signature":          0.40,
    "display_name_spoofing":      1.20,

    # ── URL signals (0-1, some are ratios clamped to 1.0) ────────────────
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
    "urgency_score":              0.80,   # already 0-1 from semantic_track
    "fear_score":                 0.90,   # already 0-1
    "curiosity_score":            0.60,   # already 0-1
    "total_phishing_keywords":    0.05,   # 0-1 ratio (clamped)
    "unique_phishing_keywords":   0.15,   # 0-1 ratio (clamped)
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
    "cape_malscore_norm":         2.50,   # 0-1 (CAPE 0-10 ÷ 10)
    "cape_behavior_risk":         2.00,   # 0-1
    "cape_network_contact":       0.80,   # 0-1 ratio
    "cape_dropped_files":         1.00,   # 0-1 ratio
    "cape_process_injection":     2.50,   # 0-1 flag

    # ── Model outputs (0-1 risk_probability) ─────────────────────────────
    "header_model_risk":          1.50,   # 0-1 from header.py
    "body_model_risk":            1.80,   # 0-1 from body.py

    # ── Routing signals ───────────────────────────────────────────────────
    "excessive_received_hops":    0.40,
    "received_from_suspicious":   0.60,
}


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DecisionResult:
    action:             str                    # "DROP" | "PEND" | "DELIVER"
    composite_score_01: float                  # 0.0–1.0  (internal)
    composite_score_10: float                  # 0.0–10.0 (display)
    confidence:         str                    # "HIGH" | "MEDIUM" | "LOW"
    triggered_rules:    list  = field(default_factory=list)
    score_breakdown:    dict  = field(default_factory=dict)
    contradiction:      dict  = field(default_factory=dict)
    explanation:        str   = ""
    timestamp:          str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "action":             self.action,
            "composite_score":    round(self.composite_score_10, 3),  # display
            "composite_score_01": round(self.composite_score_01, 4),  # internal
            "confidence":         self.confidence,
            "triggered_rules":    self.triggered_rules,
            "score_breakdown":    self.score_breakdown,
            "contradiction":      self.contradiction,
            "explanation":        self.explanation,
            "timestamp":          self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Signal extractor  — all outputs 0.0–1.0
# ═══════════════════════════════════════════════════════════════════════════
def _extract_signals(semantic_meta, header_result, body_result, attach_result) -> dict:
    """
    Flatten all model outputs + semantic metadata into a single signal dict.
    Every value MUST be in [0.0, 1.0].

    Key normalisation points:
      - header_model_risk  : already 0-1 (risk_probability)
      - body_model_risk    : already 0-1 (risk_probability)
      - cape_malscore_norm : CAPE 0-10  → ÷ 10 → 0-1
      - cape_behavior_risk : already 0-1 (from sandbox_client._parse_report)
      - keyword counts     : clamped ratios
    """
    s = semantic_meta
    h = header_result
    b = body_result
    a = attach_result

    # Model outputs — already 0-1
    h_risk = float(h.get("risk_probability", 0.5))
    b_risk = float(b.get("risk_probability", 0.5))

    # Attachment & sandbox signals
    cape_malscore    = 0.0   # will be 0-1 after ÷10
    cape_behavior    = 0.0   # 0-1
    cape_net         = 0.0   # 0-1
    cape_dropped     = 0.0   # 0-1
    cape_inject      = 0.0   # 0-1 flag
    win_exec         = 0.0
    lin_exec         = 0.0
    script_att       = 0.0
    office_macro     = 0.0
    archive_att      = 0.0
    double_ext       = 0.0

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

        if ext in WIN_EXEC_EXT:  win_exec     = 1.0
        if ext in SCRIPT_EXT:   script_att    = 1.0
        if ext in MACRO_EXT:    office_macro  = 1.0
        if ext in ARCHIVE_EXT:  archive_att   = max(archive_att, 0.5)
        if ext in LINUX_EXT:    lin_exec      = 1.0

        for cv in att.get("cape_verdicts", []):
            # CAPE malscore is 0-10 — NORMALISE to 0-1
            ms = float(cv.get("malscore", 0)) / 10.0
            cape_malscore = max(cape_malscore, min(ms, 1.0))

            # behavior_risk already 0-1 from sandbox_client._parse_report
            cape_behavior = max(cape_behavior,
                                min(float(cv.get("behavior_risk", 0)), 1.0))

            net_hosts  = len(cv.get("network", {}).get("hosts", []))
            cape_net   = min(1.0, net_hosts / 5.0)

            dropped    = int(cv.get("dropped_files", 0))
            cape_dropped = min(1.0, dropped / 3.0)

            behs = [beh.lower() for beh in cv.get("behaviors", [])]
            if "process_injection" in behs:
                cape_inject = 1.0

        # Redirect chain risk — already 0-1 from investigate_redirect
        for rd in att.get("redirect_chain", []):
            cape_behavior = max(cape_behavior,
                                min(float(rd.get("risk_score", 0)), 1.0))

    # URL signals — ratios clamped to 0-1
    url_ip       = float(s.get("url_has_ip", 0))                        # 0/1 flag
    url_mismatch = min(1.0, int(s.get("url_mismatch_count", 0)) / 3.0)  # 0-1
    url_enc      = min(1.0, int(s.get("url_encoded_count",  0)) / 5.0)  # 0-1
    url_tld      = min(1.0, int(s.get("url_suspicious_tlds",0)) / 3.0)  # 0-1

    # Keyword density — 0-1 ratios
    total_kw  = min(1.0, int(s.get("total_phishing_keywords",  0)) / 20.0)
    unique_kw = min(1.0, int(s.get("unique_phishing_keywords", 0)) / 10.0)
    money_sym = min(1.0, int(s.get("total_money_symbols",      0)) / 5.0)

    # Exclamation marks — 0-1 ratio
    subj_excl = min(1.0, int(s.get("subject_exclamation", 0)) / 3.0)

    # Hops
    hops       = int(s.get("received_hops", 0))
    excess_hops = 1.0 if hops > 10 else 0.0

    return {
        # Auth — 0/1 flags
        "spf_fail":                float(s.get("spf_fail",  0)),
        "dkim_fail":               float(s.get("dkim_fail", 0)),
        "dmarc_fail":              float(s.get("dmarc_fail",0)),
        "spf_softfail":            0.0,

        # Header — 0/1 flags and 0-1 ratios
        "domain_mismatch":         float(s.get("domain_mismatch",       0)),
        "suspicious_tld_sender":   float(s.get("suspicious_tld_sender", 0)),
        "has_numeric_in_domain":   float(s.get("has_numeric_in_domain", 0)),
        "date_is_future":          float(s.get("date_is_future",        0)),
        "missing_message_id":      float(1 - int(s.get("has_message_id", 1))),
        "missing_date":            float(1 - int(s.get("has_date",       1))),
        "no_dkim_signature":       float(1 - int(s.get("has_dkim",       1))),
        "display_name_spoofing":   float(s.get("domain_mismatch",        0)),

        # URL — 0-1
        "url_has_ip":              url_ip,
        "url_mismatch":            url_mismatch,
        "url_encoded_excess":      url_enc,
        "url_suspicious_tld":      url_tld,
        "url_shortener":           0.0,
        "url_redirect_chain_risk": cape_behavior if float(a.get("risk_probability",0)) > 0 else 0.0,

        # Content — 0/1 flags; urgency/fear/curiosity already 0-1
        "has_script":              float(s.get("has_script",         0)),
        "has_iframe":              float(s.get("has_iframe",         0)),
        "has_eval":                float(s.get("has_eval",           0)),
        "has_unescape":            float(s.get("has_unescape",       0)),
        "has_data_uri":            float(s.get("has_data_uri",       0)),
        "has_form_with_password":  float(s.get("has_input_password", 0)),
        "has_form":                float(s.get("has_form",           0)),
        "obfuscated_chars":        float(s.get("obfuscated_chars",   0)),
        "urgency_score":           min(1.0, float(s.get("urgency_score",   0))),
        "fear_score":              min(1.0, float(s.get("fear_score",      0))),
        "curiosity_score":         min(1.0, float(s.get("curiosity_score", 0))),
        "total_phishing_keywords": total_kw,
        "unique_phishing_keywords":unique_kw,
        "money_symbols":           money_sym,
        "subject_all_caps":        float(s.get("subject_all_caps", 0)),
        "subject_exclamation":     subj_excl,

        # Attachment & sandbox — ALL 0-1
        "attachment_win_executable": win_exec,
        "attachment_script":         script_att,
        "attachment_office_macro":   office_macro,
        "attachment_archive":        archive_att,
        "attachment_double_ext":     double_ext,
        "attachment_linux_exec":     lin_exec,
        "cape_malscore_norm":        cape_malscore,    # ÷10 applied above
        "cape_behavior_risk":        cape_behavior,    # already 0-1
        "cape_network_contact":      cape_net,
        "cape_dropped_files":        cape_dropped,
        "cape_process_injection":    cape_inject,

        # Models — already 0-1 risk_probability
        "header_model_risk":         h_risk,
        "body_model_risk":           b_risk,

        # Routing
        "excessive_received_hops":   excess_hops,
        "received_from_suspicious":  0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Weighted scoring  — all in 0.0–1.0, composite_01 = weighted_sum / total_w
# ═══════════════════════════════════════════════════════════════════════════
def _compute_score(signals: dict) -> tuple:
    """
    Returns (composite_01, composite_10, breakdown).

    composite_01 = weighted_sum / total_weight  → 0.0–1.0
    composite_10 = composite_01 × 10            → 0.0–10.0  (display only)

    Because every signal is 0-1 and weights are multipliers:
      max possible weighted_sum = total_weight  (all signals = 1.0)
    Dividing by total_weight normalises the result back to 0-1.
    """
    total_weight = sum(WEIGHTS.values())
    weighted_sum = 0.0
    breakdown    = {}

    for signal, weight in WEIGHTS.items():
        value = float(signals.get(signal, 0.0))
        # Safety clamp — signals must be 0-1
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
# Deterministic rule engine  — all comparisons in 0.0–1.0 space
# ═══════════════════════════════════════════════════════════════════════════
def _evaluate_rules(signals, composite_01, semantic_meta,
                    attach_result, header_result, body_result) -> tuple:
    """
    All threshold comparisons use 0-1 values consistently.
    CAPE thresholds: T_CAPE_DROP=0.70, T_CAPE_PEND=0.40  (normalised).
    Model thresholds: direct risk_probability comparison (already 0-1).
    """
    drop_reasons = []
    pend_reasons = []

    # h_risk, b_risk already 0-1 from model outputs
    h_risk = float(header_result.get("risk_probability", 0.5))
    b_risk = float(body_result.get("risk_probability",   0.5))

    # cape_malscore_norm already 0-1 (normalised in _extract_signals)
    cape_01 = float(signals.get("cape_malscore_norm", 0.0))

    # ── HARD DROP RULES ──────────────────────────────────────────────────
    # R-D01: CAPE malscore > 0.70 (≡ 7.0/10)
    if cape_01 > T_CAPE_DROP:
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
            f"R-D04: Absolute model consensus — header={h_risk:.3f} body={b_risk:.3f}"
        )

    # R-D05: IP URL + credential form
    if signals.get("url_has_ip", 0) > 0 and \
       signals.get("has_form_with_password", 0) > 0:
        drop_reasons.append(
            "R-D05: IP-based URL + password input form (credential harvesting)")

    # R-D06: Double extension executable
    if signals.get("attachment_double_ext",      0) > 0 and \
       signals.get("attachment_win_executable",  0) > 0:
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

    # R-D09: domain mismatch + suspicious TLD + SPF fail
    if (signals.get("domain_mismatch",      0) > 0 and
            signals.get("suspicious_tld_sender", 0) > 0 and
            signals.get("spf_fail",             0) > 0):
        drop_reasons.append(
            "R-D09: Domain mismatch + suspicious TLD + SPF fail")

    # R-D10: CAPE dropped files + network contact
    if signals.get("cape_dropped_files",    0) > 0 and \
       signals.get("cape_network_contact",  0) > 0:
        drop_reasons.append(
            "R-D10: Sandbox dropped files AND contacted external hosts")

    # R-D11: IP URL + body model > 0.85
    if signals.get("url_has_ip", 0) > 0 and b_risk > 0.85:
        drop_reasons.append(
            f"R-D11: IP-based URL + high body risk ({b_risk:.3f})")

    # R-D12: 3+ URL anchor mismatches
    if int(semantic_meta.get("url_mismatch_count", 0)) >= 3:
        drop_reasons.append(
            f"R-D12: {semantic_meta.get('url_mismatch_count')} URL anchor mismatches")

    # ── HARD PEND RULES ──────────────────────────────────────────────────
    # R-P01: CAPE malscore in suspicious range (0.40–0.70)
    if T_CAPE_PEND < cape_01 <= T_CAPE_DROP:
        pend_reasons.append(
            f"R-P01: CAPE malscore {cape_01*10:.1f}/10 in suspicious range (4.0–7.0)")

    # R-P02: Body high, header clean
    if b_risk > 0.70 and h_risk < 0.35:
        pend_reasons.append(
            f"R-P02: Body malicious ({b_risk:.3f}) but header clean ({h_risk:.3f})")

    # R-P03: Header high, body clean
    if h_risk > 0.70 and b_risk < 0.35:
        pend_reasons.append(
            f"R-P03: Header suspicious ({h_risk:.3f}) but body clean ({b_risk:.3f})")

    # R-P04: Executable + no authentication at all
    if (signals.get("attachment_win_executable", 0) > 0 and
            signals.get("spf_fail",  0) == 0 and
            signals.get("dkim_fail", 0) == 0 and
            int(semantic_meta.get("has_dkim", 0)) == 0):
        pend_reasons.append(
            "R-P04: Executable attachment from sender with no email auth")

    # R-P05: Office macro + any auth failure
    if signals.get("attachment_office_macro", 0) > 0 and (
            signals.get("spf_fail", 0) > 0 or signals.get("dkim_fail", 0) > 0):
        pend_reasons.append(
            "R-P05: Office macro attachment + auth failure")

    # R-P06: Archive + high body risk (> 0.65)
    if signals.get("attachment_archive", 0) > 0 and b_risk > 0.65:
        pend_reasons.append(
            f"R-P06: Archive attachment + high body risk ({b_risk:.3f})")

    # R-P07: Future date + domain mismatch
    if signals.get("date_is_future",   0) > 0 and \
       signals.get("domain_mismatch",  0) > 0:
        pend_reasons.append("R-P07: Future timestamp + domain mismatch")

    # R-P08: iframe + form
    if signals.get("has_iframe", 0) > 0 and signals.get("has_form", 0) > 0:
        pend_reasons.append("R-P08: <iframe> + <form> in email body")

    # R-P09: Urgency + fear + keyword density
    if (float(semantic_meta.get("urgency_score",            0)) > 0.60 and
            float(semantic_meta.get("fear_score",           0)) > 0.60 and
            int(semantic_meta.get("unique_phishing_keywords",0)) >= 5):
        pend_reasons.append(
            "R-P09: High urgency + fear + 5+ phishing keywords")

    # R-P10: IP URL + no Message-ID
    if signals.get("url_has_ip",       0) > 0 and \
       int(semantic_meta.get("has_message_id", 1)) == 0:
        pend_reasons.append(
            "R-P10: IP-based URL + missing Message-ID (bulk tool)")

    # R-P11: Sandbox contacted external hosts
    if signals.get("cape_network_contact", 0) > 0:
        pend_reasons.append(
            "R-P11: Sandbox attachment contacted external hosts")

    # R-P12: Script attachment
    if signals.get("attachment_script", 0) > 0:
        pend_reasons.append(
            "R-P12: Script file attachment (.js/.vbs/.ps1/.hta)")

    # R-P13: Suspicious TLD + urgency
    if (signals.get("suspicious_tld_sender",      0) > 0 and
            float(semantic_meta.get("urgency_score", 0)) > 0.40):
        pend_reasons.append(
            "R-P13: Suspicious sender TLD + urgency language")

    # R-P14: Excessive routing hops
    if signals.get("excessive_received_hops", 0) > 0:
        pend_reasons.append(
            f"R-P14: Excessive routing hops ({semantic_meta.get('received_hops', 0)})")

    # R-P15: Linux sandbox suspicious range
    cape_linux_01 = 0.0
    for att in attach_result.get("attachments", []):
        for cv in att.get("cape_verdicts", []):
            if cv.get("platform") == "linux":
                # CAPE malscore 0-10 → normalise to 0-1
                cape_linux_01 = max(
                    cape_linux_01,
                    min(float(cv.get("malscore", 0)) / 10.0, 1.0)
                )
    if T_CAPE_PEND < cape_linux_01 <= T_CAPE_DROP:
        pend_reasons.append(
            f"R-P15: Linux sandbox malscore {cape_linux_01*10:.1f}/10 in suspicious range")

    # ── Compile ──────────────────────────────────────────────────────────
    all_rules = drop_reasons + pend_reasons
    if drop_reasons:
        return all_rules, "DROP"
    if pend_reasons:
        return all_rules, "PEND"
    return all_rules, None


# ═══════════════════════════════════════════════════════════════════════════
# Contradiction detector  — compares 0-1 values directly
# ═══════════════════════════════════════════════════════════════════════════
def _detect_contradiction(header_result, body_result, attach_result) -> dict:
    h_risk  = float(header_result.get("risk_probability", 0.5))
    b_risk  = float(body_result.get("risk_probability",   0.5))
    a_risk  = float(attach_result.get("risk_probability", 0.0))
    has_att = attach_result.get("has_attachments", 0)

    contradictions = []
    gap = abs(h_risk - b_risk)

    # Gap > 0.45 on 0-1 scale (≡ 4.5 points on 0-10 scale)
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
def _calculate_confidence(composite_01, rules, contradiction) -> str:
    if contradiction["has_contradiction"] and contradiction["count"] >= 2:
        return "LOW"
    if len(rules) >= 2 and not contradiction["has_contradiction"]:
        return "HIGH"
    if (composite_01 >= T_DROP or composite_01 < 0.30) and len(rules) >= 1:
        return "HIGH"
    if abs(composite_01 - T_PEND_LO) < 0.05 or abs(composite_01 - T_DROP) < 0.05:
        return "LOW"
    return "MEDIUM"


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def decide(semantic_meta, header_result, body_result, attach_result) -> DecisionResult:
    """
    Main entry point called by mail_filter.py.

    All inputs use 0.0–1.0 risk_probability values.
    Composite score is computed internally in 0-1, reported in both scales.
    """
    # 1. Extract normalised signals (all 0-1)
    signals = _extract_signals(semantic_meta, header_result,
                               body_result, attach_result)

    # 2. Compute weighted composite (returns 0-1 and 0-10)
    composite_01, composite_10, breakdown = _compute_score(signals)

    # 3. Deterministic rules (all thresholds in 0-1 space)
    triggered_rules, forced_action = _evaluate_rules(
        signals, composite_01, semantic_meta,
        attach_result, header_result, body_result,
    )

    # 4. Contradiction detection (0-1 comparisons)
    contradiction = _detect_contradiction(header_result, body_result, attach_result)

    # 5. Final arbiter
    if forced_action == "DROP":
        action = "DROP"
    elif forced_action == "PEND":
        action = "DROP" if composite_01 >= T_DROP else "PEND"
    else:
        # Score-only path
        if composite_01 >= T_DROP:
            action = "DROP"
        elif composite_01 >= T_PEND_LO:
            action = "PEND"
        else:
            # Below PEND — check for high-severity contradiction
            if (contradiction["has_contradiction"] and
                    any(c["severity"] == "HIGH" for c in contradiction["details"])):
                action = "PEND"
                triggered_rules.append(
                    "R-C01: High-severity model contradiction → escalated to PEND"
                )
            else:
                action = "DELIVER"

    # 6. Confidence
    confidence = _calculate_confidence(composite_01, triggered_rules, contradiction)

    # 7. Explanation  — all values shown in their natural scale
    h_risk = float(header_result.get("risk_probability", 0.5))
    b_risk = float(body_result.get("risk_probability",   0.5))
    a_risk = float(attach_result.get("risk_probability", 0.0))

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
    )


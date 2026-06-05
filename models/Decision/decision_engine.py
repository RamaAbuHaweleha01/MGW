#!/usr/bin/env python3
"""
~/MGW/models/Decision/decision_engine.py
═══════════════════════════════════════════════════════════════════════════════
Comprehensive Decision Engine for the MGW Phishing Defence Gateway.

Architecture:
  1. Deterministic Rule Engine  — hard DROP/PEND triggers, no scoring
  2. Weighted Scoring Engine    — per-signal adaptive weights → composite score
  3. Contradiction Detector     — flags stark model disagreements → PEND
  4. Final Decision Arbiter     — fuses rules + score → DROP / PEND / DELIVER

Score scale: 0.0 – 10.0  (maps from 0.0 – 1.0 risk probabilities × 10)
  DROP    : composite ≥ 7.5   OR  any hard-DROP rule fires
  PEND    : composite 4.5 – 7.4  OR  any PEND rule fires
  DELIVER : composite < 4.5   AND  no rules fired

All cases and rules are documented inline.
Returns a DecisionResult object consumed by mail_filter.py.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("decision_engine")

# ═══════════════════════════════════════════════════════════════════════════
# Score thresholds
# ═══════════════════════════════════════════════════════════════════════════
THRESHOLD_DROP    = 7.5   # composite score → instant DROP
THRESHOLD_PEND_HI = 7.4   # composite score upper boundary for PEND
THRESHOLD_PEND_LO = 4.5   # composite score lower boundary for PEND
THRESHOLD_DELIVER = 4.5   # below this → DELIVER (if no rules fired)

# Cape attachment malscore thresholds (CAPE uses 0-10 scale directly)
CAPE_DROP_THRESHOLD = 7.0
CAPE_PEND_THRESHOLD = 4.0

# ═══════════════════════════════════════════════════════════════════════════
# Adaptive signal weights
# Each weight is applied to a normalised 0-1 signal before summing.
# Weights are tuned to reflect real-world phishing campaign patterns.
# ═══════════════════════════════════════════════════════════════════════════
WEIGHTS = {
    # ── Authentication failures ──────────────────────────────────────────
    "spf_fail":                   1.50,   # SPF hard fail — strong sender forgery signal
    "dkim_fail":                  1.20,   # DKIM fail — body/header tampering
    "dmarc_fail":                 1.30,   # DMARC fail — policy violation
    "spf_softfail":               0.60,   # SPF ~all — weaker but suspicious

    # ── Header / sender signals ──────────────────────────────────────────
    "domain_mismatch":            1.40,   # From ≠ Reply-To or Return-Path
    "suspicious_tld_sender":      1.10,   # Sender on .tk/.xyz/etc
    "has_numeric_in_domain":      0.50,   # e.g. pay0pal.com
    "date_is_future":             0.80,   # Timestamp manipulation
    "missing_message_id":         0.60,   # Unusual for legitimate mailers
    "missing_date":               0.50,
    "no_dkim_signature":          0.40,   # Not signed at all
    "display_name_spoofing":      1.20,   # Display name ≠ email domain

    # ── URL / link signals ───────────────────────────────────────────────
    "url_has_ip":                 1.60,   # http://1.2.3.4/ — almost never legit
    "url_mismatch":               1.50,   # Anchor text ≠ href destination
    "url_encoded_excess":         0.80,   # Heavy URL encoding / obfuscation
    "url_suspicious_tld":         1.00,   # Links to .tk/.xyz/etc
    "url_shortener":              0.70,   # Redirect chain obscures destination
    "url_redirect_chain_risk":    1.20,   # Redirect lands on suspicious page

    # ── Content / body signals ───────────────────────────────────────────
    "has_script":                 1.40,   # <script> in email body
    "has_iframe":                 1.30,   # <iframe> embedding
    "has_eval":                   1.60,   # eval() — code execution attempt
    "has_unescape":               1.40,   # unescape() — obfuscation
    "has_data_uri":               1.20,   # data:base64 — inline payload
    "has_form_with_password":     1.70,   # Credential harvesting form
    "has_form":                   0.80,   # Any HTML form
    "obfuscated_chars":           0.90,   # HTML entity abuse
    "urgency_score":              0.80,   # "Act immediately / expire / deadline"
    "fear_score":                 0.90,   # "Account blocked / terminated"
    "curiosity_score":            0.60,   # "You've won / selected"
    "total_phishing_keywords":    0.05,   # Per keyword (capped)
    "unique_phishing_keywords":   0.15,   # Per unique keyword (capped)
    "money_symbols":              0.40,   # $$$, wire transfer language
    "subject_all_caps":           0.50,   # ALL CAPS subject
    "subject_exclamation":        0.30,   # Exclamation marks

    # ── Attachment / sandbox signals ─────────────────────────────────────
    "attachment_win_executable":  2.00,   # .exe/.msi/.bat/.cmd/.ps1/.vbs/.lnk
    "attachment_script":          1.60,   # .js/.vbs/.ps1 as attachment
    "attachment_office_macro":    1.40,   # .docm/.xlsm/.pptm
    "attachment_archive":         0.80,   # .zip/.rar/.7z (may contain payload)
    "attachment_double_ext":      1.80,   # invoice.pdf.exe — extension spoofing
    "attachment_linux_exec":      1.50,   # .sh/.elf/.bin on Linux sandbox
    "cape_malscore_norm":         2.50,   # Normalised CAPE malscore (0-1)
    "cape_behavior_risk":         2.00,   # CAPE behavioral signature risk
    "cape_network_contact":       0.80,   # Contacted external hosts during analysis
    "cape_dropped_files":         1.00,   # Dropped additional files
    "cape_process_injection":     2.50,   # Process injection detected

    # ── Model outputs ────────────────────────────────────────────────────
    "header_model_risk":          1.50,   # Header model risk probability
    "body_model_risk":            1.80,   # Body model risk probability (higher weight)

    # ── Received / routing signals ───────────────────────────────────────
    "excessive_received_hops":    0.40,   # > 10 hops — unusual routing
    "received_from_suspicious":   0.60,   # Relay from known bad ASN/TLD
}


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DecisionResult:
    action:          str            # "DROP" | "PEND" | "DELIVER"
    composite_score: float          # 0.0 – 10.0
    confidence:      str            # "HIGH" | "MEDIUM" | "LOW"
    triggered_rules: list[str]      = field(default_factory=list)
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    contradiction:   dict[str, Any] = field(default_factory=dict)
    explanation:     str            = ""
    timestamp:       str            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "action":          self.action,
            "composite_score": round(self.composite_score, 4),
            "confidence":      self.confidence,
            "triggered_rules": self.triggered_rules,
            "score_breakdown": self.score_breakdown,
            "contradiction":   self.contradiction,
            "explanation":     self.explanation,
            "timestamp":       self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Signal extractor
# ═══════════════════════════════════════════════════════════════════════════
def _extract_signals(
    semantic_meta: dict,
    header_result: dict,
    body_result:   dict,
    attach_result: dict,
) -> dict[str, float]:
    """
    Flatten all model outputs and semantic metadata into a single
    normalised signal dict  (values clamped to 0.0 – 1.0).
    """
    s = semantic_meta
    h = header_result
    b = body_result
    a = attach_result

    h_risk = float(h.get("risk_probability", 0.5))
    b_risk = float(b.get("risk_probability", 0.5))

    # ── Attachment signals ────────────────────────────────────────────────
    att_max_risk  = float(a.get("risk_probability", 0.0))
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

    WIN_EXEC_EXT  = {".exe",".msi",".bat",".cmd",".lnk",".dll"}
    SCRIPT_EXT    = {".ps1",".vbs",".js",".wsf",".hta"}
    MACRO_EXT     = {".docm",".xlsm",".pptm",".rtf"}
    ARCHIVE_EXT   = {".zip",".rar",".7z",".iso",".img"}
    LINUX_EXT     = {".sh",".elf",".bin",".run",".deb",".rpm"}

    for att in a.get("attachments", []):
        fname = (att.get("filename") or "").lower()
        import os
        ext   = os.path.splitext(fname)[1]
        # Double extension check  invoice.pdf.exe
        base  = os.path.splitext(fname)[0]
        if os.path.splitext(base)[1] in {".pdf",".doc",".xls",".png",".jpg"}:
            double_ext = 1.0

        if ext in WIN_EXEC_EXT:  win_exec    = 1.0
        if ext in SCRIPT_EXT:   script_att   = 1.0
        if ext in MACRO_EXT:    office_macro = 1.0
        if ext in ARCHIVE_EXT:  archive_att  = max(archive_att, 0.5)
        if ext in LINUX_EXT:    lin_exec     = 1.0

        for cv in att.get("cape_verdicts", []):
            ms = float(cv.get("malscore", 0)) / 10.0
            cape_malscore = max(cape_malscore, ms)
            cape_behavior = max(cape_behavior,
                                float(cv.get("behavior_risk", 0)))
            net_hosts = len(cv.get("network", {}).get("hosts", []))
            cape_net  = min(1.0, net_hosts / 5.0)
            dropped   = int(cv.get("dropped_files", 0))
            cape_dropped = min(1.0, dropped / 3.0)
            # Process injection from behavior names
            behs = [b.lower() for b in cv.get("behaviors", [])]
            if "process_injection" in behs:
                cape_inject = 1.0

        # Redirect chain risk
        for rd in att.get("redirect_chain", []):
            cape_behavior = max(cape_behavior,
                                float(rd.get("risk_score", 0)))

    # ── URL signals ───────────────────────────────────────────────────────
    url_count    = int(s.get("url_count", 0))
    url_ip       = float(s.get("url_has_ip", 0))
    url_mismatch = min(1.0, int(s.get("url_mismatch_count", 0)) / 3.0)
    url_enc      = min(1.0, int(s.get("url_encoded_count", 0)) / 5.0)
    url_tld      = min(1.0, int(s.get("url_suspicious_tlds", 0)) / 3.0)

    # ── Subject signals ───────────────────────────────────────────────────
    subj_caps   = float(s.get("subject_all_caps",    0))
    subj_excl   = min(1.0, int(s.get("subject_exclamation", 0)) / 3.0)

    # ── Keyword signals ───────────────────────────────────────────────────
    total_kw    = min(1.0, int(s.get("total_phishing_keywords",  0)) / 20.0)
    unique_kw   = min(1.0, int(s.get("unique_phishing_keywords", 0)) / 10.0)
    money_sym   = min(1.0, int(s.get("total_money_symbols", 0)) / 5.0)

    # ── Received hops ─────────────────────────────────────────────────────
    hops        = int(s.get("received_hops", 0))
    excess_hops = 1.0 if hops > 10 else 0.0

    return {
        # Auth
        "spf_fail":               float(s.get("spf_fail",  0)),
        "dkim_fail":              float(s.get("dkim_fail", 0)),
        "dmarc_fail":             float(s.get("dmarc_fail",0)),
        "spf_softfail":           0.0,   # placeholder — extend if parser adds it
        # Header
        "domain_mismatch":        float(s.get("domain_mismatch",       0)),
        "suspicious_tld_sender":  float(s.get("suspicious_tld_sender", 0)),
        "has_numeric_in_domain":  float(s.get("has_numeric_in_domain", 0)),
        "date_is_future":         float(s.get("date_is_future", 0)),
        "missing_message_id":     float(1 - s.get("has_message_id", 1)),
        "missing_date":           float(1 - s.get("has_date", 1)),
        "no_dkim_signature":      float(1 - s.get("has_dkim", 1)),
        "display_name_spoofing":  float(s.get("domain_mismatch", 0)),
        # URL
        "url_has_ip":             url_ip,
        "url_mismatch":           url_mismatch,
        "url_encoded_excess":     url_enc,
        "url_suspicious_tld":     url_tld,
        "url_shortener":          0.0,   # placeholder
        "url_redirect_chain_risk":cape_behavior if att_max_risk > 0 else 0.0,
        # Content
        "has_script":             float(s.get("has_script",         0)),
        "has_iframe":             float(s.get("has_iframe",         0)),
        "has_eval":               float(s.get("has_eval",           0)),
        "has_unescape":           float(s.get("has_unescape",       0)),
        "has_data_uri":           float(s.get("has_data_uri",       0)),
        "has_form_with_password": float(s.get("has_input_password", 0)),
        "has_form":               float(s.get("has_form",           0)),
        "obfuscated_chars":       float(s.get("obfuscated_chars",   0)),
        "urgency_score":          float(s.get("urgency_score",      0)),
        "fear_score":             float(s.get("fear_score",         0)),
        "curiosity_score":        float(s.get("curiosity_score",    0)),
        "total_phishing_keywords": total_kw,
        "unique_phishing_keywords":unique_kw,
        "money_symbols":          money_sym,
        "subject_all_caps":       subj_caps,
        "subject_exclamation":    subj_excl,
        # Attachments
        "attachment_win_executable":  win_exec,
        "attachment_script":          script_att,
        "attachment_office_macro":    office_macro,
        "attachment_archive":         archive_att,
        "attachment_double_ext":      double_ext,
        "attachment_linux_exec":      lin_exec,
        "cape_malscore_norm":         cape_malscore,
        "cape_behavior_risk":         cape_behavior,
        "cape_network_contact":       cape_net,
        "cape_dropped_files":         cape_dropped,
        "cape_process_injection":     cape_inject,
        # Models
        "header_model_risk":      h_risk,
        "body_model_risk":        b_risk,
        # Routing
        "excessive_received_hops": excess_hops,
        "received_from_suspicious": 0.0,   # placeholder
    }


# ═══════════════════════════════════════════════════════════════════════════
# Weighted scoring engine
# ═══════════════════════════════════════════════════════════════════════════
def _compute_score(signals: dict[str, float]) -> tuple[float, dict]:
    """
    Compute weighted composite score in range 0 – 10.
    Returns (composite_score, breakdown_dict).
    """
    total_weight     = sum(WEIGHTS.values())
    weighted_sum     = 0.0
    breakdown        = {}

    for signal, weight in WEIGHTS.items():
        value = signals.get(signal, 0.0)
        contribution = value * weight
        weighted_sum += contribution
        if contribution > 0.01:   # only log meaningful contributions
            breakdown[signal] = round(contribution, 4)

    # Normalise to 0-10 scale
    # Raw weighted_sum can exceed 10 for worst-case emails — clamp
    max_possible = total_weight          # if every signal = 1.0
    normalised   = (weighted_sum / max_possible) * 10.0
    composite    = round(min(normalised, 10.0), 4)

    return composite, breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic rule engine
# ═══════════════════════════════════════════════════════════════════════════
def _evaluate_rules(
    signals: dict[str, float],
    composite: float,
    semantic_meta: dict,
    attach_result:  dict,
    header_result:  dict,
    body_result:    dict,
) -> tuple[list[str], str | None]:
    """
    Evaluate all deterministic rules.
    Returns (triggered_rules, forced_action | None).
    forced_action overrides the score-based decision.
    """
    rules:   list[str] = []
    drop_reasons: list[str] = []
    pend_reasons: list[str] = []

    h_risk = float(header_result.get("risk_probability", 0.5))
    b_risk = float(body_result.get("risk_probability",   0.5))

    # ── HARD DROP RULES ───────────────────────────────────────────────────
    # R-D01: CAPE malscore > 7 (malware confirmed by sandbox)
    if signals["cape_malscore_norm"] * 10 > CAPE_DROP_THRESHOLD:
        drop_reasons.append("R-D01: CAPE malscore > 7 (sandbox confirmed malware)")

    # R-D02: Process injection detected by CAPE
    if signals["cape_process_injection"] > 0:
        drop_reasons.append("R-D02: Process injection detected by sandbox")

    # R-D03: Executable attachment + ALL auth failures
    if (signals["attachment_win_executable"] > 0 and
            signals["spf_fail"] > 0 and
            signals["dkim_fail"] > 0 and
            signals["dmarc_fail"] > 0):
        drop_reasons.append(
            "R-D03: Executable attachment + SPF+DKIM+DMARC all failed")

    # R-D04: Absolute model consensus — both models > 0.9
    if h_risk > 0.90 and b_risk > 0.90:
        drop_reasons.append(
            f"R-D04: Absolute model consensus — header={h_risk:.3f} body={b_risk:.3f}")

    # R-D05: IP address in URL + credential harvesting form
    if signals["url_has_ip"] > 0 and signals["has_form_with_password"] > 0:
        drop_reasons.append(
            "R-D05: IP-based URL + password input form (credential harvesting)")

    # R-D06: Double extension attachment + executable
    if signals["attachment_double_ext"] > 0 and signals["attachment_win_executable"] > 0:
        drop_reasons.append("R-D06: Double-extension executable (e.g. invoice.pdf.exe)")

    # R-D07: eval() or unescape() in body + script tag
    if signals["has_eval"] > 0 and signals["has_script"] > 0:
        drop_reasons.append("R-D07: JavaScript eval() + <script> — code execution attempt")

    # R-D08: Data URI + script tag
    if signals["has_data_uri"] > 0 and signals["has_script"] > 0:
        drop_reasons.append("R-D08: data: URI + <script> — inline payload delivery")

    # R-D09: Domain mismatch + suspicious TLD + SPF fail
    if (signals["domain_mismatch"] > 0 and
            signals["suspicious_tld_sender"] > 0 and
            signals["spf_fail"] > 0):
        drop_reasons.append(
            "R-D09: Domain mismatch + suspicious TLD + SPF fail (spoofed sender)")

    # R-D10: CAPE dropped files + network contact
    if signals["cape_dropped_files"] > 0 and signals["cape_network_contact"] > 0:
        drop_reasons.append(
            "R-D10: Sandbox dropped files AND contacted external hosts")

    # R-D11: URL to IP + body model > 0.85
    if signals["url_has_ip"] > 0 and b_risk > 0.85:
        drop_reasons.append(
            f"R-D11: IP-based URL + high body risk ({b_risk:.3f})")

    # R-D12: Three or more URL mismatches (sophisticated phishing)
    if int(semantic_meta.get("url_mismatch_count", 0)) >= 3:
        drop_reasons.append(
            f"R-D12: {semantic_meta.get('url_mismatch_count')} URL anchor mismatches")

    # ── HARD PEND RULES ───────────────────────────────────────────────────
    # R-P01: CAPE malscore 4-7 (suspicious but not confirmed)
    cape_score = signals["cape_malscore_norm"] * 10
    if CAPE_PEND_THRESHOLD < cape_score <= CAPE_DROP_THRESHOLD:
        pend_reasons.append(
            f"R-P01: CAPE malscore in suspicious range ({cape_score:.1f}/10)")

    # R-P02: Model contradiction — body risk high, header clean
    contradiction_gap = b_risk - h_risk
    if b_risk > 0.70 and h_risk < 0.35:
        pend_reasons.append(
            f"R-P02: Stark contradiction — clean header ({h_risk:.3f}) "
            f"but malicious body ({b_risk:.3f})")

    # R-P03: Model contradiction — header high, body clean
    if h_risk > 0.70 and b_risk < 0.35:
        pend_reasons.append(
            f"R-P03: Stark contradiction — suspicious header ({h_risk:.3f}) "
            f"but clean body ({b_risk:.3f})")

    # R-P04: Executable attachment from unverified sender (no auth)
    if (signals["attachment_win_executable"] > 0 and
            signals["spf_fail"] == 0 and
            signals["dkim_fail"] == 0 and
            semantic_meta.get("has_dkim", 0) == 0):
        pend_reasons.append(
            "R-P04: Executable attachment from sender with no email authentication")

    # R-P05: Office macro attachment + any auth failure
    if signals["attachment_office_macro"] > 0 and (
            signals["spf_fail"] > 0 or signals["dkim_fail"] > 0):
        pend_reasons.append(
            "R-P05: Office macro attachment + auth failure")

    # R-P06: Archive attachment + high body risk
    if signals["attachment_archive"] > 0 and b_risk > 0.65:
        pend_reasons.append(
            f"R-P06: Archive attachment + high body risk ({b_risk:.3f})")

    # R-P07: Future date + domain mismatch (timestamp manipulation)
    if signals["date_is_future"] > 0 and signals["domain_mismatch"] > 0:
        pend_reasons.append(
            "R-P07: Future timestamp + domain mismatch")

    # R-P08: iframe + form (common phishing page embedding)
    if signals["has_iframe"] > 0 and signals["has_form"] > 0:
        pend_reasons.append(
            "R-P08: <iframe> + <form> in email body")

    # R-P09: High urgency + fear + keyword density
    if (float(semantic_meta.get("urgency_score", 0)) > 0.60 and
            float(semantic_meta.get("fear_score",    0)) > 0.60 and
            int(semantic_meta.get("unique_phishing_keywords", 0)) >= 5):
        pend_reasons.append(
            "R-P09: High urgency + fear score + 5+ phishing keywords")

    # R-P10: URL to IP with no SPF verification
    if signals["url_has_ip"] > 0 and signals["spf_fail"] == 0 and \
            semantic_meta.get("has_message_id", 1) == 0:
        pend_reasons.append(
            "R-P10: IP-based URL + no Message-ID (bulk tool indicator)")

    # R-P11: CAPE sandbox contacted external hosts
    if signals["cape_network_contact"] > 0:
        pend_reasons.append(
            "R-P11: Sandbox attachment contacted external hosts during analysis")

    # R-P12: Script attachment (.js/.vbs/.ps1)
    if signals["attachment_script"] > 0:
        pend_reasons.append(
            "R-P12: Script file attachment (.js/.vbs/.ps1)")

    # R-P13: Suspicious sender TLD + urgency signals
    if signals["suspicious_tld_sender"] > 0 and \
            float(semantic_meta.get("urgency_score", 0)) > 0.40:
        pend_reasons.append(
            "R-P13: Suspicious sender TLD + urgency language")

    # R-P14: Excessive received hops (routing anomaly)
    if signals["excessive_received_hops"] > 0:
        pend_reasons.append(
            f"R-P14: Excessive routing hops ({semantic_meta.get('received_hops', 0)})")

    # R-P15: Linux executable to linux sandbox — sandbox PEND range
    cape_score_linux = 0.0
    for att in attach_result.get("attachments", []):
        for cv in att.get("cape_verdicts", []):
            if cv.get("platform") == "linux":
                cape_score_linux = max(
                    cape_score_linux,
                    float(cv.get("malscore", 0))
                )
    if CAPE_PEND_THRESHOLD < cape_score_linux <= CAPE_DROP_THRESHOLD:
        pend_reasons.append(
            f"R-P15: Linux sandbox malscore in suspicious range ({cape_score_linux:.1f})")

    # ── Compile result ────────────────────────────────────────────────────
    rules = drop_reasons + pend_reasons
    if drop_reasons:
        return rules, "DROP"
    if pend_reasons:
        return rules, "PEND"
    return rules, None


# ═══════════════════════════════════════════════════════════════════════════
# Contradiction detector
# ═══════════════════════════════════════════════════════════════════════════
def _detect_contradiction(
    header_result: dict,
    body_result:   dict,
    attach_result: dict,
) -> dict:
    """
    Detect stark disagreements between models.
    Returns a dict describing any contradictions found.
    """
    h_risk    = float(header_result.get("risk_probability", 0.5))
    b_risk    = float(body_result.get("risk_probability",   0.5))
    a_risk    = float(attach_result.get("risk_probability", 0.0))
    has_att   = attach_result.get("has_attachments", 0)

    contradictions = []
    gap = abs(h_risk - b_risk)

    if gap > 0.45:
        contradictions.append({
            "type":        "header_body_gap",
            "description": (
                f"Header risk ({h_risk:.3f}) and body risk ({b_risk:.3f}) "
                f"diverge by {gap:.3f} — possible social engineering"
            ),
            "severity": "HIGH" if gap > 0.55 else "MEDIUM",
        })

    if has_att and a_risk == 0.0 and (h_risk > 0.65 or b_risk > 0.65):
        contradictions.append({
            "type":        "clean_attachment_suspicious_content",
            "description": (
                "Attachment is clean but email content is highly suspicious — "
                "possible decoy attachment or unanalysed payload type"
            ),
            "severity": "MEDIUM",
        })

    if has_att and a_risk > 0.70 and h_risk < 0.30 and b_risk < 0.30:
        contradictions.append({
            "type":        "clean_content_malicious_attachment",
            "description": (
                "Email content appears clean but attachment is high-risk — "
                "targeted attack with benign-looking message body"
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
def _calculate_confidence(
    composite: float,
    rules: list[str],
    contradiction: dict,
    signals: dict,
) -> str:
    """
    HIGH   — multiple independent signals agree, rules fired
    MEDIUM — score-driven decision, limited corroboration
    LOW    — contradictions present or borderline score
    """
    if contradiction["has_contradiction"] and contradiction["count"] >= 2:
        return "LOW"
    if len(rules) >= 2 and not contradiction["has_contradiction"]:
        return "HIGH"
    if (composite > THRESHOLD_DROP or composite < 3.0) and len(rules) >= 1:
        return "HIGH"
    if abs(composite - THRESHOLD_PEND_LO) < 0.5 or \
       abs(composite - THRESHOLD_DROP) < 0.5:
        return "LOW"
    return "MEDIUM"


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
def decide(
    semantic_meta: dict,
    header_result: dict,
    body_result:   dict,
    attach_result: dict,
) -> DecisionResult:
    """
    Main entry point called by mail_filter.py.

    Parameters
    ----------
    semantic_meta : output of mail_filter.semantic_track()
    header_result : output of header.py  {risk_probability, engine, ...}
    body_result   : output of body.py    {risk_probability, engine, ...}
    attach_result : output of attachment_track()

    Returns
    -------
    DecisionResult with action, score, rules, and explanation.
    """
    # 1. Extract normalised signals
    signals = _extract_signals(
        semantic_meta, header_result, body_result, attach_result
    )

    # 2. Compute weighted composite score (0-10)
    composite, breakdown = _compute_score(signals)

    # 3. Evaluate deterministic rules
    triggered_rules, forced_action = _evaluate_rules(
        signals, composite, semantic_meta,
        attach_result, header_result, body_result,
    )

    # 4. Detect model contradictions
    contradiction = _detect_contradiction(
        header_result, body_result, attach_result
    )

    # 5. Final decision arbiter
    if forced_action == "DROP":
        action = "DROP"
    elif forced_action == "PEND":
        # PEND from rules — but if score is also in DROP zone, escalate
        action = "DROP" if composite >= THRESHOLD_DROP else "PEND"
    else:
        # Score-only decision
        if composite >= THRESHOLD_DROP:
            action = "DROP"
        elif composite >= THRESHOLD_PEND_LO:
            action = "PEND"
        else:
            # Below PEND threshold — but check contradiction
            if contradiction["has_contradiction"] and \
               any(c["severity"] == "HIGH" for c in contradiction["details"]):
                action = "PEND"
                triggered_rules.append(
                    "R-C01: High-severity model contradiction → escalated to PEND"
                )
            else:
                action = "DELIVER"

    # 6. Confidence
    confidence = _calculate_confidence(
        composite, triggered_rules, contradiction, signals
    )

    # 7. Human-readable explanation
    h_risk = float(header_result.get("risk_probability", 0.5))
    b_risk = float(body_result.get("risk_probability",   0.5))
    a_risk = float(attach_result.get("risk_probability", 0.0))
    explanation_parts = [
        f"Composite score: {composite:.2f}/10  "
        f"(header={h_risk:.4f} body={b_risk:.4f} attach={a_risk:.4f})",
    ]
    if triggered_rules:
        explanation_parts.append(
            f"Rules fired ({len(triggered_rules)}): "
            + "; ".join(triggered_rules[:3])
            + ("..." if len(triggered_rules) > 3 else "")
        )
    if contradiction["has_contradiction"]:
        explanation_parts.append(
            f"Contradiction detected: {contradiction['count']} model disagreement(s)"
        )
    explanation_parts.append(f"Decision: {action}  Confidence: {confidence}")

    return DecisionResult(
        action          = action,
        composite_score = composite,
        confidence      = confidence,
        triggered_rules = triggered_rules,
        score_breakdown = breakdown,
        contradiction   = contradiction,
        explanation     = " | ".join(explanation_parts),
    )

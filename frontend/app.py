"""CerviRisk-MM Streamlit frontend.

A clinical-style UI for the CerviRisk-MM API. Runs as a separate process
and calls the existing FastAPI endpoints over HTTP. The frontend adds no
backend behavior — it only renders what the API returns.

Architecture:
    Browser  →  Streamlit (port 8501)  →  FastAPI (port 8000)  →  Model

Run with:
    streamlit run frontend/app.py
or:
    .\\run.ps1 ui
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CerviRisk-MM",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Brand palette — derived directly from the logo
# ---------------------------------------------------------------------------
BRAND_NAVY        = "#0F2B4C"   # deep navy (CerviRisk text, left shield)
BRAND_NAVY_BAR    = "#3B7CC9"   # brighter navy for chart bars on dark themes
BRAND_MAGENTA     = "#A12F77"   # rose-magenta (MM text, right shield)
BRAND_MAGENTA_BAR = "#D04A9A"   # brighter magenta for chart bars on dark themes
BRAND_AMBER       = "#D97706"   # accent / moderate tier
BRAND_MUTED       = "#6B7280"   # secondary text

# Clinical traffic-light mapping using brand colors:
#   navy   = calm / low-risk / no-drift / risk-decreasing factor
#   amber  = caution / moderate / minor-drift
#   magenta= attention / high-risk / significant-drift / risk-increasing factor
TIER_COLORS = {
    "low":      BRAND_NAVY,
    "moderate": BRAND_AMBER,
    "high":     BRAND_MAGENTA,
}

SEVERITY_COLORS = {
    "none":        BRAND_NAVY,
    "minor":       BRAND_AMBER,
    "significant": BRAND_MAGENTA,
}

# Three canonical sample patients
SAMPLE_PATIENTS = {
    "Low-risk profile": {
        "Age": 24, "Number of sexual partners": 1, "First sexual intercourse": 19,
        "Num of pregnancies": 0, "Smokes": 0, "Smokes (years)": 0,
        "Hormonal Contraceptives": 0, "IUD": 0,
        "STDs": 0, "STDs:HPV": 0, "Dx:HPV": 0,
        "Hinselmann": 0, "Schiller": 0, "Citology": 0,
        # Multi-modal context: no HPV detected → no strain assigned, PRS at population baseline
        "host_prs": 1.05, "matched_super_pop": "AMR",
    },
    "Moderate-risk profile": {
        "Age": 38, "Number of sexual partners": 4, "First sexual intercourse": 16,
        "Num of pregnancies": 2, "Smokes": 1, "Smokes (years)": 12,
        "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 6,
        "IUD": 0, "STDs": 1, "STDs:HPV": 1, "Dx:HPV": 1,
        "Hinselmann": 0, "Schiller": 1, "Citology": 0,
        # Multi-modal context: HPV-positive → HPV16 assigned, PRS elevated
        "assigned_hpv_strain": "HPV16", "strain_carcinogenicity": 0.95,
        "host_prs": 1.42, "matched_super_pop": "AMR",
    },
    "High-risk profile": {
        "Age": 45, "Number of sexual partners": 6, "First sexual intercourse": 15,
        "Num of pregnancies": 3, "Smokes": 1, "Smokes (years)": 25,
        "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 15,
        "IUD": 1, "IUD (years)": 5,
        "STDs": 1, "STDs (number)": 2, "STDs:HPV": 1, "Dx:HPV": 1, "Dx:CIN": 1,
        "Hinselmann": 1, "Schiller": 1, "Citology": 1,
        # Multi-modal context: confirmed HPV16, high PRS
        "assigned_hpv_strain": "HPV16", "strain_carcinogenicity": 0.95,
        "host_prs": 1.65, "matched_super_pop": "AMR",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def api_get(url: str, path: str, timeout: int = 5) -> tuple[bool, dict | None, str | None]:
    """GET request returning (ok, body, error_message)."""
    try:
        r = requests.get(f"{url}{path}", timeout=timeout)
        if r.status_code == 200:
            return True, r.json(), None
        return False, None, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        return False, None, "Connection refused — is the API running?"
    except requests.exceptions.Timeout:
        return False, None, "Timed out"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def api_post(url: str, path: str, body: dict, timeout: int = 10) -> tuple[bool, dict | None, str | None]:
    """POST request returning (ok, body, error_message)."""
    try:
        r = requests.post(f"{url}{path}", json=body, timeout=timeout)
        if r.status_code == 200:
            return True, r.json(), None
        return False, None, f"HTTP {r.status_code}: {r.text[:300]}"
    except requests.exceptions.ConnectionError:
        return False, None, "Connection refused — is the API running?"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def render_tier_badge(tier: str, probability: float) -> None:
    """Big colored tier badge with probability."""
    color = TIER_COLORS.get(tier, "#6b7280")
    st.markdown(f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            font-family: sans-serif;
        ">
            <div style="font-size: 1.1em; opacity: 0.9;">PREDICTED RISK</div>
            <div style="font-size: 3.5em; font-weight: 700; line-height: 1.0;">
                {probability*100:.1f}%
            </div>
            <div style="font-size: 1.2em; letter-spacing: 0.15em; margin-top: 8px;">
                TIER: {tier.upper()}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_severity_badge(severity: str, score: float, method: str = "PSI") -> None:
    """Drift severity badge."""
    color = SEVERITY_COLORS.get(severity, "#6b7280")
    st.markdown(f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-family: sans-serif;
        ">
            <div style="font-size: 0.9em; opacity: 0.9;">{method} score</div>
            <div style="font-size: 2.0em; font-weight: 700; line-height: 1.0;">
                {score:.3f}
            </div>
            <div style="font-size: 0.95em; letter-spacing: 0.1em; margin-top: 4px;">
                {severity.upper()}
            </div>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo at the top of the sidebar — graceful fallback if the asset is missing
    _logo_path = Path(__file__).parent / "assets" / "logo.png"
    if _logo_path.exists():
        st.image(str(_logo_path), use_container_width=True)
    else:
        st.title("CerviRisk-MM")
    st.caption("Cervical cancer risk prediction — research prototype")

    api_url = st.text_input("API URL", value=DEFAULT_API_URL, help="Base URL of the FastAPI service")

    # Connection probe
    ok, health, err = api_get(api_url, "/health", timeout=2)
    if ok and health and health.get("model_loaded"):
        st.success(f"API connected · model loaded")
    elif ok:
        st.warning("API up · model NOT loaded")
        st.caption("Run: `.\\run.ps1 fit` or `.\\run.ps1 train`")
    else:
        st.error(f"API unreachable")
        st.caption(f"_{err}_")
        st.caption("Start it with: `.\\run.ps1 serve`")

    st.divider()

    # Patient picker
    st.subheader("Patient")
    profile_choice = st.radio(
        "Pick a profile or build a custom patient:",
        list(SAMPLE_PATIENTS) + ["Custom..."],
        index=1,  # default to Moderate
    )

    if profile_choice == "Custom...":
        st.caption("Adjust below; missing fields are imputed by the model.")
        with st.expander("Demographics", expanded=True):
            age = st.slider("Age", 13, 85, 35)
            partners = st.slider("Number of sexual partners", 0, 30, 3)
            first_sex = st.slider("First sexual intercourse (age)", 10, 35, 17)
            pregnancies = st.slider("Number of pregnancies", 0, 10, 1)
        with st.expander("Lifestyle"):
            smokes = st.checkbox("Smokes", value=False)
            smokes_years = st.slider("Smokes (years)", 0, 50, 0, disabled=not smokes)
            contraceptive = st.checkbox("Hormonal contraceptives", value=False)
            contraceptive_years = st.slider(
                "Hormonal contraceptives (years)", 0, 30, 0,
                disabled=not contraceptive,
            )
            iud = st.checkbox("IUD", value=False)
        with st.expander("STD history"):
            std = st.checkbox("Any STD history", value=False)
            std_hpv = st.checkbox("HPV exposure (STDs:HPV)", value=False)
            dx_hpv = st.checkbox("Prior HPV diagnosis (Dx:HPV)", value=False)
        with st.expander("Prior screening tests"):
            hinselmann = st.checkbox("Hinselmann positive", value=False)
            schiller = st.checkbox("Schiller positive", value=False)
            citology = st.checkbox("Cytology positive", value=False)
        with st.expander("Multi-modal context (genetics + virology)"):
            # HPV strain — populated from the project's strain catalog
            STRAIN_OPTIONS = ["(none — no HPV detected)",
                              "HPV16", "HPV18", "HPV31", "HPV33", "HPV45",
                              "HPV52", "HPV58", "OTHER_HR_HPV", "LOW_RISK_HPV"]
            STRAIN_CARCINOGENICITY = {
                "HPV16": 0.95, "HPV18": 0.85,
                "HPV31": 0.70, "HPV33": 0.70, "HPV45": 0.70,
                "HPV52": 0.65, "HPV58": 0.65,
                "OTHER_HR_HPV": 0.50, "LOW_RISK_HPV": 0.05,
            }
            strain_pick = st.selectbox(
                "Assigned HPV strain",
                STRAIN_OPTIONS,
                index=0,
                help="If a patient sample was genotyped, pick the dominant "
                     "strain. The model uses both the strain identity and its "
                     "carcinogenicity score (IARC classification).",
            )
            assigned_strain = None if strain_pick.startswith("(none") else strain_pick

            # Ancestry — populated from 1000 Genomes super-populations
            ANCESTRY_OPTIONS = [
                ("AMR", "AMR — Admixed American (default for UCI Caracas cohort)"),
                ("EUR", "EUR — European"),
                ("AFR", "AFR — African"),
                ("EAS", "EAS — East Asian"),
                ("SAS", "SAS — South Asian"),
            ]
            ancestry_label = st.selectbox(
                "Matched ancestry (1000 Genomes)",
                [label for _, label in ANCESTRY_OPTIONS],
                index=0,
                help="The 1000G super-population whose allele frequencies were "
                     "used to compute this patient's polygenic risk score. "
                     "Defaults to AMR for the UCI Caracas cohort.",
            )
            super_pop = next(code for code, label in ANCESTRY_OPTIONS
                              if label == ancestry_label)

            # PRS — slider for the polygenic risk score
            host_prs = st.slider(
                "Polygenic risk score (PRS)",
                min_value=0.0, max_value=3.0, value=1.10, step=0.05,
                help=("Sum of effect-allele dosages × log-OR weights across 11 "
                       "cervical cancer GWAS loci. Population means: AFR ≈ 1.16, "
                       "AMR ≈ 1.12, EUR ≈ 1.09, SAS ≈ 1.04 (from your 1000G panel)."),
            )

        patient = {
            "Age": age,
            "Number of sexual partners": partners,
            "First sexual intercourse": first_sex,
            "Num of pregnancies": pregnancies,
            "Smokes": int(smokes),
            "Smokes (years)": smokes_years if smokes else 0,
            "Hormonal Contraceptives": int(contraceptive),
            "Hormonal Contraceptives (years)": contraceptive_years if contraceptive else 0,
            "IUD": int(iud),
            "STDs": int(std),
            "STDs:HPV": int(std_hpv),
            "Dx:HPV": int(dx_hpv),
            "Hinselmann": int(hinselmann),
            "Schiller": int(schiller),
            "Citology": int(citology),
            # Multi-modal extras
            "host_prs": host_prs,
            "matched_super_pop": super_pop,
        }
        if assigned_strain:
            patient["assigned_hpv_strain"] = assigned_strain
            patient["strain_carcinogenicity"] = STRAIN_CARCINOGENICITY.get(assigned_strain, 0.0)
    else:
        patient = SAMPLE_PATIENTS[profile_choice]
        with st.expander("Patient features", expanded=False):
            st.json(patient)


# ---------------------------------------------------------------------------
# Main — three tabs
# ---------------------------------------------------------------------------
st.title("CerviRisk-MM")
st.caption(
    "Multi-modal cervical cancer risk prediction · "
    "research prototype, not a medical device"
)

tab_predict, tab_drift, tab_model = st.tabs([
    "Predict",
    "Drift detection",
    "Model info",
])


# ---------- Tab 1: Predict --------------------------------------------------
def _render_modality_bar(label: str, filled: int, total: int, color: str, sublabel: str = "") -> None:
    """Mini horizontal progress bar for one of the three modalities."""
    pct = (filled / total * 100) if total > 0 else 0
    st.markdown(f"""
        <div style="margin-bottom: 4px;">
            <div style="
                display: flex;
                justify-content: space-between;
                font-family: sans-serif;
                font-size: 0.85em;
                color: #4b5563;
                margin-bottom: 2px;
            ">
                <span><b>{label}</b></span>
                <span style="color: {color};">{sublabel}</span>
            </div>
            <div style="
                background: #e5e7eb;
                height: 10px;
                border-radius: 5px;
                overflow: hidden;
            ">
                <div style="
                    background: {color};
                    height: 100%;
                    width: {pct:.0f}%;
                "></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def _resolve_raw_value(feature_id: str, display_name: str, patient_dict: dict) -> str:
    """Map a SHAP feature back to the patient's original (pre-preprocessing) value.

    Handles three cases:
      1. Plain numeric: 'num__Age' → patient['Age'] → "45"
      2. One-hot category: 'cat__assigned_hpv_strain_HPV16' → "yes" or "no"
      3. Unknown: fall back to "—"
    """
    name = feature_id
    if "__" in name:
        name = name.split("__", 1)[1]

    # One-hot expansion like "assigned_hpv_strain_HPV16"
    for col in ("assigned_hpv_strain", "matched_super_pop"):
        if name.startswith(col + "_"):
            value = name[len(col) + 1:]
            return "yes" if patient_dict.get(col) == value else "no"

    raw = patient_dict.get(name)
    if raw is None:
        return "—"
    # Binary fields show as yes/no for readability
    if isinstance(raw, (int, float)) and raw in (0, 1):
        # but only if the feature itself is binary-ish
        if name in {"Smokes", "Hormonal Contraceptives", "IUD",
                     "STDs", "STDs:HPV", "STDs:HIV",
                     "Dx:Cancer", "Dx:CIN", "Dx:HPV",
                     "Hinselmann", "Schiller", "Citology"}:
            return "yes" if int(raw) == 1 else "no"
    if isinstance(raw, float):
        return f"{raw:.2f}"
    return str(raw)


def _render_contribution_row(rank: int, item: dict, max_abs: float,
                              patient_dict: dict) -> None:
    """One line in the feature-contribution panel — theme-agnostic colors."""
    contrib = item["contribution"]
    direction = item["direction"]
    width_pct = (abs(contrib) / max_abs * 100) if max_abs > 0 else 0
    bar_color = BRAND_MAGENTA_BAR if direction == "increases_risk" else BRAND_NAVY_BAR
    text_color = BRAND_MAGENTA_BAR if direction == "increases_risk" else BRAND_NAVY_BAR
    arrow = "↑" if direction == "increases_risk" else "↓" if direction == "decreases_risk" else "→"
    name = item["display_name"]
    raw_value = _resolve_raw_value(item.get("feature", ""), name, patient_dict)
    # CSS variables in Streamlit's theme:
    #   --text-color is white in dark mode, dark gray in light mode
    # We use it directly so contrast is correct in both themes.
    st.markdown(f"""
        <div style="
            display: grid;
            grid-template-columns: 28px 240px 1fr 130px 90px;
            align-items: center;
            font-family: sans-serif;
            font-size: 0.92em;
            padding: 6px 0;
            border-bottom: 1px solid rgba(128,128,128,0.18);
        ">
            <div style="color: var(--text-color); opacity: 0.5;">#{rank}</div>
            <div style="color: var(--text-color); font-weight: 500;">{name}</div>
            <div style="background: rgba(128,128,128,0.18); border-radius: 4px;
                         height: 14px; position: relative;
                         margin: 0 16px;">
                <div style="
                    background: {bar_color};
                    height: 100%;
                    width: {width_pct:.0f}%;
                    border-radius: 4px;
                "></div>
            </div>
            <div style="color: var(--text-color); opacity: 0.85;
                         text-align: right; padding-right: 12px;
                         font-size: 0.9em;">
                input: <b style="opacity:1.0;">{raw_value}</b>
            </div>
            <div style="color: {text_color}; text-align: right; font-weight: 700;">
                {arrow} {contrib:+.3f}
            </div>
        </div>
    """, unsafe_allow_html=True)


with tab_predict:
    if st.button("Run prediction", type="primary", use_container_width=True):
        with st.spinner("Calling /predict/cervical-risk..."):
            ok, body, err = api_post(api_url, "/predict/cervical-risk", patient)

        if not ok:
            st.error(f"Prediction failed: {err}")
        else:
            audit = body.get("audit") or {}
            explanation = body.get("explanation") or {}

            # ===========================================================
            # ROW 1: Patient summary card + Big risk badge
            # ===========================================================
            col_summary, col_risk = st.columns([2, 1])

            with col_summary:
                host = audit.get("host_genetics", {}) or {}
                viral = audit.get("hpv_viral", {}) or {}
                clinical = audit.get("clinical", {}) or {}

                summary_lines = []
                age = patient.get("Age", "?")
                pop = host.get("matched_super_pop") or "—"
                summary_lines.append(f"**{age} yo** patient · ancestry: **{pop}**")

                cl_text = []
                if patient.get("Smokes"):
                    cl_text.append(f"smoker ({patient.get('Smokes (years)', '?')}y)")
                if patient.get("Number of sexual partners"):
                    cl_text.append(f"{patient.get('Number of sexual partners')} partners")
                pos_screen = clinical.get("screening_tests_positive", 0)
                tot_screen = clinical.get("screening_tests_total", 3)
                if pos_screen > 0:
                    cl_text.append(f"{pos_screen}/{tot_screen} screening tests positive")

                summary_lines.append("**Clinical** — " + (", ".join(cl_text) if cl_text else "no flagged risk factors"))

                if host.get("polygenic_risk_score") is not None:
                    prs = host['polygenic_risk_score']
                    pct = host.get("percentile_in_training_distribution")
                    interp = host.get("interpretation", "—")
                    pct_str = f" ({pct}th percentile)" if pct is not None else ""
                    summary_lines.append(f"**Host genetics** — PRS {prs}{pct_str} · {interp}")

                if viral.get("assigned_strain"):
                    strain = viral["assigned_strain"]
                    carc = viral.get("carcinogenicity")
                    iarc = viral.get("iarc_classification") or "—"
                    summary_lines.append(
                        f"**Viral** — assigned {strain}"
                        + (f" · carcinogenicity {carc}" if carc is not None else "")
                        + f" · {iarc}"
                    )

                st.markdown("##### Patient assessment")
                for line in summary_lines:
                    st.markdown(f"  {line}")
                st.caption(f"Model: `{body.get('model_version', '—')}` · "
                            f"data_status: `{body.get('data_status', '—')}`")

            with col_risk:
                render_tier_badge(body["tier"], body["risk_probability"])
                rec = {
                    "low":      "Routine screening, next cycle.",
                    "moderate": "Expedited follow-up; repeat cytology in 6 months.",
                    "high":     "Refer for diagnostic biopsy.",
                }.get(body["tier"], "—")
                st.markdown(f"<div style='text-align:center; margin-top:8px; "
                            f"font-weight:600;'>{rec}</div>",
                            unsafe_allow_html=True)

            st.divider()

            # ===========================================================
            # ROW 2: Feature contributions panel
            # ===========================================================
            st.markdown("##### Feature contributions to this classification")
            st.caption(
                "Top features pushing this prediction toward HIGH (red, ↑) or "
                "LOW (green, ↓) risk. Computed via SHAP TreeExplainer on the "
                "deployed model — same direction and magnitude the model used."
            )

            contributors = explanation.get("top_contributors") or []
            if not contributors:
                # Show the actual reason for failure so we can debug
                method = explanation.get("method", "unavailable")
                err = explanation.get("error")
                clf_type = explanation.get("classifier_type")
                if err:
                    st.warning(
                        f"**Feature contributions unavailable** — {method}\n\n"
                        f"_Reason:_ `{err}`"
                        + (f"\n\n_Classifier:_ `{clf_type}`" if clf_type else "")
                    )
                else:
                    st.info(
                        "Per-prediction feature contributions are not available."
                    )
            else:
                max_abs = max(abs(c["contribution"]) for c in contributors)
                for i, c in enumerate(contributors, 1):
                    _render_contribution_row(i, c, max_abs, patient)
                st.caption(
                    f"_Method: {explanation.get('method', 'SHAP')} · "
                    f"{explanation.get('n_features_in_model', '?')} features in model · "
                    "the **input** column shows the raw patient value submitted; "
                    "the **±** column shows the model's log-odds contribution._"
                )

            st.divider()

            # ===========================================================
            # ROW 3: Multi-modal evidence integration
            # ===========================================================
            st.markdown("##### Multi-modal evidence integration")
            st.caption(
                "How each evidence modality contributes to the overall picture. "
                "The deployed `triage + xgb` model integrates all three in the "
                "same probability."
            )
            col_clin, col_host, col_viral = st.columns(3)

            with col_clin:
                pos = clinical.get("screening_tests_positive", 0)
                tot = clinical.get("screening_tests_total", 3)
                rf = clinical.get("high_risk_factors_count", 0)
                rf_max = clinical.get("high_risk_factors_max", 4)
                cl_label = ("HIGH" if pos >= 2 else "MODERATE"
                            if pos == 1 or rf >= 2 else "NORMAL")
                cl_color = (BRAND_MAGENTA if cl_label == "HIGH"
                            else BRAND_AMBER if cl_label == "MODERATE"
                            else BRAND_NAVY)
                st.markdown(f"**CLINICAL** &nbsp; <span style='color:{cl_color}; "
                            f"font-weight:700; letter-spacing:0.05em;'>{cl_label}</span>",
                            unsafe_allow_html=True)
                _render_modality_bar(
                    f"Screening tests",
                    pos, tot, cl_color, f"{pos}/{tot} positive"
                )
                _render_modality_bar(
                    f"Lifestyle risk factors",
                    rf, rf_max, cl_color, f"{rf}/{rf_max}"
                )

            with col_host:
                pct = host.get("percentile_in_training_distribution")
                interp = host.get("interpretation", "not_provided")
                hcolor = (BRAND_MAGENTA if interp == "elevated"
                          else BRAND_AMBER if interp == "average"
                          else BRAND_NAVY if interp == "lower"
                          else BRAND_MUTED)
                hlabel = interp.upper().replace("_", " ")
                st.markdown(f"**HOST GENETICS** &nbsp; <span style='color:{hcolor}; "
                            f"font-weight:700; letter-spacing:0.05em;'>{hlabel}</span>",
                            unsafe_allow_html=True)
                if pct is not None:
                    _render_modality_bar(
                        f"PRS percentile",
                        pct, 100, hcolor, f"{pct}th percentile",
                    )
                else:
                    st.caption("_PRS percentile unavailable (baseline not loaded)_")
                pop = host.get("matched_super_pop") or "—"
                st.caption(f"Matched ancestry: **{pop}** (1000 Genomes panel)")

            with col_viral:
                strain = viral.get("assigned_strain")
                carc = viral.get("carcinogenicity")
                iarc = viral.get("iarc_classification") or "—"
                vlabel = ("HIGH-RISK" if carc and carc >= 0.7
                          else "MODERATE" if carc and carc >= 0.3
                          else "LOW-RISK" if carc is not None
                          else "NO STRAIN")
                vcolor = (BRAND_MAGENTA if vlabel == "HIGH-RISK"
                          else BRAND_AMBER if vlabel == "MODERATE"
                          else BRAND_NAVY if vlabel == "LOW-RISK"
                          else BRAND_MUTED)
                st.markdown(f"**VIRAL** &nbsp; <span style='color:{vcolor}; "
                            f"font-weight:700; letter-spacing:0.05em;'>{vlabel}</span>",
                            unsafe_allow_html=True)
                if strain:
                    if carc is not None:
                        _render_modality_bar(
                            f"Carcinogenicity",
                            int(carc * 100), 100, vcolor,
                            f"{carc:.2f}",
                        )
                    st.caption(f"Strain: **{strain}** · {iarc}")
                else:
                    st.caption("_No HPV strain assigned to this patient_")

            st.divider()

            # Collapsible technical details
            with st.expander("Full API response (JSON)"):
                st.json(body)
            with st.expander("Patient features submitted"):
                st.json(patient)

    else:
        st.info(
            "Configure the patient in the sidebar, then click **Run prediction**. "
            "Sample profiles are available; you can also build a custom patient."
        )


# ---------- Tab 2: Drift ----------------------------------------------------
with tab_drift:
    # ---- LIVE NCBI section (top, with auto-refresh) -----------------------
    st.markdown("#### LIVE — direct from NCBI E-utilities")
    st.caption(
        "Pulls HPV sequence deposits from the last 30 days, computes the "
        "strain distribution, and compares against the published de Sanjosé "
        "2010 baseline. Auto-refreshes every minute. NCBI is hit at most "
        "once every 5 minutes (server-side cache)."
    )

    # Auto-refresh — every 60 seconds
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="ncbi_live_refresh")
        _autorefresh_ok = True
    except ImportError:
        _autorefresh_ok = False
        st.caption("(install `streamlit-autorefresh` to enable auto-polling — "
                   "manual refresh below still works)")

    col_a, col_b = st.columns([1, 4])
    with col_a:
        force = st.button("Refresh now")
    with col_b:
        st.caption("Click to force-bypass the 5-minute server cache.")

    ok, live, err = api_get(api_url, f"/drift/strain/live?force_refresh={'true' if force else 'false'}", timeout=15)
    if not ok:
        st.error(f"Live fetch failed: {err}")
    elif live and not live.get("ncbi_meta", {}).get("ok", True):
        st.error(f"NCBI unavailable: {live['ncbi_meta'].get('error', 'unknown')}")
        st.caption(f"Query was: `{live['ncbi_meta'].get('query', '?')}`")
    elif live:
        cache_source = live.get("cache_source", "?")
        cache_age = live.get("cache_age_seconds", 0)
        n_seqs = live.get("n_sequences_typed", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sequences (last 30d)", n_seqs)
        col2.metric("Cache source",
                    "FRESH" if cache_source == "fresh" else "CACHED",
                    delta=f"{int(cache_age)}s old" if cache_source == "cached" else "just fetched")
        col3.metric("PSI vs baseline", f"{live['psi_score']:.3f}")
        col4.metric("Severity", live["psi_severity"].upper())

        # Big action badge
        action = live.get("action_recommended", "no_action")
        action_color = {
            "retrain":   BRAND_MAGENTA,
            "monitor":   BRAND_AMBER,
            "no_action": BRAND_NAVY,
        }.get(action, "#6b7280")
        st.markdown(f"""
            <div style="
                background-color: {action_color};
                color: white;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
                font-family: sans-serif;
                margin: 16px 0;
            ">
                <div style="font-size: 0.85em; opacity: 0.9;">RECOMMENDED ACTION</div>
                <div style="font-size: 1.6em; font-weight: 700; letter-spacing: 0.05em;">
                    {action.upper().replace('_', ' ')}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Side-by-side bar chart
        baseline_dist = live.get("baseline_distribution", {})
        current_dist = live.get("current_distribution", {})
        if baseline_dist and current_dist:
            strains = sorted(set(baseline_dist) | set(current_dist),
                              key=lambda k: -baseline_dist.get(k, 0))
            df_compare = pd.DataFrame({
                "Baseline (de Sanjosé 2010)": [baseline_dist.get(k, 0) for k in strains],
                "LIVE (NCBI last 30d)":        [current_dist.get(k, 0) for k in strains],
            }, index=strains)
            st.bar_chart(df_compare)

        # Top contributors
        contribs = live.get("top_contributors", {})
        if contribs:
            with st.expander("Top drift contributors"):
                rows = []
                for cat, v in contribs.items():
                    arrow = "↑" if current_dist.get(cat, 0) > baseline_dist.get(cat, 0) else "↓"
                    rows.append({"strain": cat, "direction": arrow,
                                 "contribution": round(v, 4)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

        chi_p = live.get("chi_square_pvalue", 1.0)
        chi_drift = chi_p < 0.05
        st.caption(
            f"Chi-square goodness-of-fit p-value: **{chi_p:.4f}** "
            f"({'drift detected' if chi_drift else 'no drift'}). "
            f"Last fetched at: {live.get('fetched_at', '—')}."
        )

    st.divider()

    # ---- Static baseline section (existing) -------------------------------
    st.markdown("#### Saved training baseline vs uploaded batch")
    st.caption(
        "Drift detection compares incoming patient batches against the "
        "training-time baseline. PSI ≥ 0.20 = significant drift, "
        "PSI ≥ 0.10 = minor, < 0.10 = no drift."
    )

    if st.button("Fetch saved baseline", use_container_width=True):
        ok, body, err = api_get(api_url, "/drift/baseline")
        if not ok:
            st.error(f"Could not load baseline: {err}")
            st.caption("Run: `.\\run.ps1 baseline`")
        else:
            st.session_state["baseline"] = body

    if "baseline" in st.session_state:
        baseline = st.session_state["baseline"]
        st.success(f"Baseline loaded · {baseline.get('n_records', '?')} reference patients")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Captured at", baseline.get("captured_at", "—")[:10])
        with col2:
            cat_count = len(baseline.get("categorical_features", {}))
            num_count = len(baseline.get("numeric_features", {}))
            st.metric("Features tracked", f"{num_count} numeric, {cat_count} categorical")

        # Visualize strain distribution if available
        strains = baseline.get("categorical_features", {}).get("assigned_hpv_strain")
        if strains and "proportions" in strains:
            st.subheader("Baseline HPV strain composition")
            df = pd.DataFrame(
                [{"strain": k, "proportion": v} for k, v in strains["proportions"].items()]
            ).sort_values("proportion", ascending=False)
            st.bar_chart(df.set_index("strain"))

    st.divider()
    st.markdown("##### Run drift check on a batch")
    st.caption(
        "Submit one or many patient records to check against the baseline. "
        "The current sidebar patient is used; click multiple times to "
        "simulate batches with this profile."
    )

    n_copies = st.slider("Number of copies to submit", 5, 200, 50,
                          help="More records → tighter drift estimate")

    if st.button("Run drift check", use_container_width=True):
        body = {"records": [patient] * n_copies}
        ok, drift_body, err = api_post(api_url, "/drift/check", body)
        if not ok:
            st.error(f"Drift check failed: {err}")
        else:
            action = drift_body["action_recommended"]
            severity_color = {"retrain": BRAND_MAGENTA,
                              "monitor": BRAND_AMBER,
                              "no_action": BRAND_NAVY}.get(action, "#6b7280")
            st.markdown(f"""
                <div style="
                    background-color: {severity_color};
                    color: white;
                    padding: 16px;
                    border-radius: 8px;
                    text-align: center;
                    font-family: sans-serif;
                    margin: 16px 0;
                ">
                    <div style="font-size: 1.0em; opacity: 0.9;">RECOMMENDED ACTION</div>
                    <div style="font-size: 2.0em; font-weight: 700; letter-spacing: 0.05em;">
                        {action.upper().replace('_', ' ')}
                    </div>
                    <div style="font-size: 0.9em; margin-top: 4px;">
                        {drift_body['n_features_with_drift']} of
                        {drift_body['n_features_checked']} features drifted
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("Per-feature drift scores")
            rows = []
            for f in drift_body.get("by_feature", []):
                rows.append({
                    "feature": f["feature"],
                    "method": f["method"].upper(),
                    "score": f["score"],
                    "severity": f["severity"],
                    "drift": "✔" if f["drift_detected"] else "—",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

            with st.expander("Full drift response"):
                st.json(drift_body)


# ---------- Tab 3: Model info -----------------------------------------------
with tab_model:
    if st.button("Refresh model info", use_container_width=True):
        ok, body, err = api_get(api_url, "/model/info")
        if not ok:
            st.error(f"Could not load model info: {err}")
        else:
            st.session_state["model_info"] = body

    info = st.session_state.get("model_info")
    if info is None:
        # auto-fetch on first load
        ok, body, err = api_get(api_url, "/model/info")
        if ok:
            info = body
            st.session_state["model_info"] = body

    if info:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Model", info.get("model_name", "—"))
            st.metric("Version", info.get("model_version", "—"))
        with col_b:
            best = info.get("best_variant") or {}
            st.metric("Best variant", f"{best.get('mode')} + {best.get('model')}"
                      if best.get("mode") else "—")
            st.metric("Eval protocol", best.get("eval", "—"))

        if best:
            st.subheader("Held-out performance")
            cols = st.columns(4)
            cols[0].metric("DEV AUPRC", f"{best.get('dev_auprc_pct', '—')}%")
            cols[1].metric("DEV AUROC", f"{best.get('dev_auroc_pct', '—')}%")
            cols[2].metric("DEV Sens", f"{best.get('dev_sensitivity_pct', '—')}%")
            cols[3].metric("DEV Spec", f"{best.get('dev_specificity_pct', '—')}%")

            cols = st.columns(4)
            cols[0].metric("TEST AUPRC",
                            f"{best.get('test_auprc_pct', '—')}% "
                            f"± {best.get('test_auprc_std_pct', '—')}%")

        st.subheader("Notes")
        for note in info.get("notes", []):
            st.write(f"- {note}")

        with st.expander("Tuned hyperparameters across all variants"):
            st.json(info.get("tuned_hyperparameters") or {})


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"_Developed at {datetime.utcnow().isoformat(timespec='seconds')} UTC. "
    f"Research prototype not a medical device._"
    f"Nabras Al-Mahrami, nabras.almahrami@ochs.edu.om"
)

"""CerviRisk-MM — Streamlit Community Cloud deployment.

Self-contained: loads the trained model directly with joblib. No FastAPI
process needed alongside. The full version with live NCBI drift monitoring
lives in frontend/app.py and is launched locally via `.\\run.ps1 ui`.

Live: https://cervirisk-mm.streamlit.app/
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup + brand palette
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CerviRisk-MM",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_NAVY        = "#0F2B4C"
BRAND_NAVY_BAR    = "#3B7CC9"
BRAND_MAGENTA     = "#A12F77"
BRAND_MAGENTA_BAR = "#D04A9A"
BRAND_AMBER       = "#D97706"
BRAND_MUTED       = "#6B7280"

TIER_COLORS = {
    "low":      BRAND_NAVY,
    "moderate": BRAND_AMBER,
    "high":     BRAND_MAGENTA,
}

DISCLAIMER = (
    "Research prototype — not a medical device. Predictions must not be "
    "used for clinical diagnosis or treatment decisions without appropriate "
    "regulatory approval and clinical validation."
)

# Deployed model metrics — these are part of the frozen v0.1 artifact,
# hardcoded for reliable display (the JSON format may differ from training run
# to training run; the cloud demo serves a fixed model with fixed numbers).
DEPLOYED_METRICS = {
    "variant": "triage + xgb (tuned)",
    "dev": {
        "auprc": 65.2, "auroc": 96.7,
        "sensitivity": 93.2, "specificity": 95.3,
        "n": 686, "positives": 44,
    },
    "test": {
        "auprc": (76.7, 26.6), "auroc": (95.4, 7.1),
        "sensitivity": (80.0, 40.0), "specificity": (96.3, 2.3),
        "n": 172, "positives": 11,
    },
}

ALL_FEATURES = [
    "Age", "Number of sexual partners", "First sexual intercourse",
    "Num of pregnancies", "Smokes", "Smokes (years)",
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)",
    "IUD", "IUD (years)",
    "STDs", "STDs (number)", "STDs:HPV", "STDs:HIV",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV",
    "Hinselmann", "Schiller", "Citology",
    "host_prs", "strain_carcinogenicity",
    "assigned_hpv_strain", "matched_super_pop",
]

SAMPLE_PATIENTS = {
    "Low-risk profile": {
        "Age": 24, "Number of sexual partners": 1, "First sexual intercourse": 19,
        "Num of pregnancies": 0, "Smokes": 0, "Smokes (years)": 0,
        "Hormonal Contraceptives": 0, "IUD": 0,
        "STDs": 0, "STDs:HPV": 0, "Dx:HPV": 0,
        "Hinselmann": 0, "Schiller": 0, "Citology": 0,
        "host_prs": 1.05, "matched_super_pop": "AMR",
    },
    "Moderate-risk profile": {
        "Age": 38, "Number of sexual partners": 4, "First sexual intercourse": 16,
        "Num of pregnancies": 2, "Smokes": 1, "Smokes (years)": 12,
        "Hormonal Contraceptives": 1, "Hormonal Contraceptives (years)": 6,
        "IUD": 0, "STDs": 1, "STDs:HPV": 1, "Dx:HPV": 1,
        "Hinselmann": 0, "Schiller": 1, "Citology": 0,
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
        "assigned_hpv_strain": "HPV16", "strain_carcinogenicity": 0.95,
        "host_prs": 1.65, "matched_super_pop": "AMR",
    },
}


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> tuple[object | None, Path | None]:
    for c in (Path("models/cervirisk_mm_v0.1.pkl"),
              Path(__file__).parent / "models" / "cervirisk_mm_v0.1.pkl"):
        if c.exists():
            return joblib.load(c), c
    return None, None


@st.cache_resource
def load_baseline() -> dict | None:
    for c in (Path("models/drift_baseline.json"),
              Path(__file__).parent / "models" / "drift_baseline.json"):
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def features_to_row(patient: dict) -> pd.DataFrame:
    return pd.DataFrame([{f: patient.get(f, None) for f in ALL_FEATURES}])


def tier_for(p: float) -> str:
    if p < 0.20: return "low"
    if p < 0.50: return "moderate"
    return "high"


def compute_shap(model, df: pd.DataFrame, top_k: int = 10) -> list[dict]:
    try:
        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        X_t = pre.transform(df)
        try:
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            feature_names = [f"f{i}" for i in range(X_t.shape[1])]

        import xgboost as xgb
        booster = clf.get_booster()
        try:
            dmat = xgb.DMatrix(X_t, feature_names=booster.feature_names)
        except Exception:
            dmat = xgb.DMatrix(X_t)
        contribs = booster.predict(dmat, pred_contribs=True)
        shap_row = contribs[0, :-1]

        ranked = []
        for raw, val, contrib in zip(feature_names, X_t[0], shap_row):
            name = raw.split("__", 1)[1] if "__" in raw else raw
            for prefix in ("assigned_hpv_strain_", "matched_super_pop_"):
                if name.startswith(prefix):
                    base = prefix.rstrip("_")
                    value = name[len(prefix):]
                    name = f"{base}: {value}"
                    break
            ranked.append({
                "name": name, "value": float(val),
                "contribution": float(contrib),
                "direction": "up" if contrib > 0 else "down",
            })
        ranked.sort(key=lambda r: -abs(r["contribution"]))
        return ranked[:top_k]
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


def resolve_raw_value(name: str, patient: dict) -> str:
    for prefix in ("assigned_hpv_strain", "matched_super_pop"):
        if name.startswith(prefix + ":"):
            wanted = name.split(": ", 1)[1]
            return "yes" if patient.get(prefix) == wanted else "no"
    raw = patient.get(name)
    if raw is None:
        return "—"
    if isinstance(raw, (int, float)) and raw in (0, 1):
        if name in {"Smokes", "Hormonal Contraceptives", "IUD",
                     "STDs", "STDs:HPV", "STDs:HIV",
                     "Dx:Cancer", "Dx:CIN", "Dx:HPV",
                     "Hinselmann", "Schiller", "Citology"}:
            return "yes" if int(raw) == 1 else "no"
    if isinstance(raw, float):
        return f"{raw:.2f}"
    return str(raw)


def render_tier_badge(tier: str, prob: float) -> None:
    color = TIER_COLORS.get(tier, BRAND_MUTED)
    st.markdown(f"""
        <div style="background:{color};color:white;padding:24px;
                     border-radius:12px;text-align:center;font-family:sans-serif;">
            <div style="font-size:1.1em;opacity:0.9;">PREDICTED RISK</div>
            <div style="font-size:3.5em;font-weight:700;line-height:1.0;">
                {prob*100:.1f}%
            </div>
            <div style="font-size:1.2em;letter-spacing:0.15em;margin-top:8px;">
                TIER: {tier.upper()}
            </div>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_path = Path("frontend/assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.title("CerviRisk-MM")
    st.caption("Multi-modal cervical cancer risk prediction · research prototype")
    st.divider()

    model, model_path = load_model()
    if model is None:
        st.error("Model artifact not found in the repo.")
        st.caption(
            "If you're running locally, train the model with: `.\\run.ps1 fit` "
            "or `.\\run.ps1 train`."
        )
        st.stop()
    st.success("Model loaded")
    st.caption(f"`{model_path}`")

    st.divider()
    profile_choice = st.radio(
        "Pick a patient profile:",
        list(SAMPLE_PATIENTS) + ["Custom..."],
        index=1,
    )

    if profile_choice == "Custom...":
        with st.expander("Demographics", expanded=True):
            age = st.slider("Age", 13, 85, 35)
            partners = st.slider("Number of sexual partners", 0, 30, 3)
            first_sex = st.slider("First sexual intercourse (age)", 10, 35, 17)
            pregnancies = st.slider("Number of pregnancies", 0, 10, 1)
        with st.expander("Lifestyle"):
            smokes = st.checkbox("Smokes")
            smokes_years = st.slider("Smokes (years)", 0, 50, 0, disabled=not smokes)
            contraceptive = st.checkbox("Hormonal contraceptives")
            iud = st.checkbox("IUD")
        with st.expander("STD history"):
            std = st.checkbox("Any STD history")
            std_hpv = st.checkbox("HPV exposure")
            dx_hpv = st.checkbox("Prior HPV diagnosis")
        with st.expander("Prior screening tests"):
            hinselmann = st.checkbox("Hinselmann positive")
            schiller = st.checkbox("Schiller positive")
            citology = st.checkbox("Cytology positive")
        with st.expander("Multi-modal context (genetics + virology)"):
            strain_pick = st.selectbox(
                "Assigned HPV strain",
                ["(none — no HPV detected)", "HPV16", "HPV18", "HPV31",
                 "HPV33", "HPV45", "HPV52", "HPV58",
                 "OTHER_HR_HPV", "LOW_RISK_HPV"],
            )
            super_pop = st.selectbox(
                "Matched ancestry (1000G)",
                ["AMR", "EUR", "AFR", "EAS", "SAS"],
            )
            host_prs = st.slider("Polygenic risk score (PRS)",
                                  0.0, 3.0, 1.10, 0.05)

        patient = {
            "Age": age, "Number of sexual partners": partners,
            "First sexual intercourse": first_sex,
            "Num of pregnancies": pregnancies,
            "Smokes": int(smokes), "Smokes (years)": smokes_years if smokes else 0,
            "Hormonal Contraceptives": int(contraceptive),
            "IUD": int(iud), "STDs": int(std),
            "STDs:HPV": int(std_hpv), "Dx:HPV": int(dx_hpv),
            "Hinselmann": int(hinselmann), "Schiller": int(schiller),
            "Citology": int(citology),
            "host_prs": host_prs, "matched_super_pop": super_pop,
        }
        if not strain_pick.startswith("(none"):
            patient["assigned_hpv_strain"] = strain_pick
            patient["strain_carcinogenicity"] = {
                "HPV16": 0.95, "HPV18": 0.85,
                "HPV31": 0.70, "HPV33": 0.70, "HPV45": 0.70,
                "HPV52": 0.65, "HPV58": 0.65,
                "OTHER_HR_HPV": 0.50, "LOW_RISK_HPV": 0.05,
            }.get(strain_pick, 0.0)
    else:
        patient = SAMPLE_PATIENTS[profile_choice]
        with st.expander("Submitted features", expanded=False):
            st.json(patient)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("CerviRisk-MM")
st.caption(
    "Multi-modal cervical cancer risk prediction · research prototype, "
    "not a medical device"
)

tab_predict, tab_drift, tab_about = st.tabs(
    ["Predict", "Drift detection", "About this model"]
)

# ============================================================================
# Tab 1: PREDICT
# ============================================================================
with tab_predict:
    if st.button("Run prediction", type="primary", use_container_width=True):
        df = features_to_row(patient)
        try:
            proba = float(model.predict_proba(df)[0, 1])
        except Exception as e:
            st.error(f"Prediction failed: {type(e).__name__}: {e}")
            st.stop()

        tier = tier_for(proba)
        col_summary, col_risk = st.columns([2, 1])
        with col_summary:
            st.subheader("Patient assessment")
            age_val = patient.get("Age", "?")
            pop = patient.get("matched_super_pop", "—")
            st.markdown(f"  **{age_val} yo** patient · ancestry: **{pop}**")
            cl = []
            if patient.get("Smokes"):
                cl.append(f"smoker ({patient.get('Smokes (years)', '?')}y)")
            screening_pos = sum(int(bool(patient.get(k, 0))) for k in
                                  ("Hinselmann", "Schiller", "Citology"))
            if screening_pos > 0:
                cl.append(f"{screening_pos}/3 screening tests positive")
            if cl:
                st.markdown(f"  **Clinical** — {', '.join(cl)}")
            if patient.get("host_prs") is not None:
                st.markdown(f"  **Host genetics** — PRS {patient['host_prs']:.2f}")
            strain = patient.get("assigned_hpv_strain")
            if strain:
                carc = patient.get("strain_carcinogenicity")
                line = f"  **Viral** — {strain}"
                if carc is not None:
                    line += f" · carcinogenicity {carc}"
                st.markdown(line)

        with col_risk:
            render_tier_badge(tier, proba)
            rec = {
                "low":      "Routine screening, next cycle.",
                "moderate": "Expedited follow-up; repeat cytology in 6 months.",
                "high":     "Refer for diagnostic biopsy.",
            }[tier]
            st.markdown(f"<div style='text-align:center;margin-top:8px;"
                        f"font-weight:600;'>{rec}</div>",
                        unsafe_allow_html=True)

        st.divider()
        st.subheader("Feature contributions")
        st.caption(
            "Top features pushing this prediction toward HIGH (magenta, ↑) or "
            "LOW (navy, ↓) risk. Computed via XGBoost built-in TreeSHAP."
        )
        contributors = compute_shap(model, df, top_k=10)
        if contributors and "error" not in contributors[0]:
            max_abs = max(abs(c["contribution"]) for c in contributors)
            for i, c in enumerate(contributors, 1):
                color = BRAND_MAGENTA_BAR if c["direction"] == "up" else BRAND_NAVY_BAR
                arrow = "↑" if c["direction"] == "up" else "↓"
                width = abs(c["contribution"]) / max_abs * 100
                raw_val = resolve_raw_value(c["name"], patient)
                st.markdown(f"""
                    <div style="display:grid;
                                 grid-template-columns:28px 220px 1fr 110px 90px;
                                 align-items:center;font-family:sans-serif;
                                 font-size:0.9em;padding:5px 0;
                                 border-bottom:1px solid rgba(128,128,128,0.15);">
                        <div style="color:var(--text-color);opacity:0.5;">#{i}</div>
                        <div style="color:var(--text-color);font-weight:500;">{c['name']}</div>
                        <div style="background:rgba(128,128,128,0.18);
                                     border-radius:4px;height:14px;margin:0 16px;">
                            <div style="background:{color};height:100%;
                                         width:{width:.0f}%;border-radius:4px;"></div>
                        </div>
                        <div style="color:var(--text-color);opacity:0.85;
                                     text-align:right;padding-right:12px;
                                     font-size:0.85em;">
                            input: <b style="opacity:1.0;">{raw_val}</b>
                        </div>
                        <div style="color:{color};text-align:right;
                                     font-weight:700;">{arrow} {c['contribution']:+.3f}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            err = contributors[0].get("error", "unknown") if contributors else "no output"
            st.warning(f"Feature contributions unavailable: `{err}`")

        st.divider()
        with st.expander("Raw output"):
            st.json({
                "risk_probability": round(proba, 4),
                "tier": tier,
                "model_version": "cervirisk_mm_v0.1",
                "data_status": "RESEARCH_PROTOTYPE",
                "disclaimer": DISCLAIMER,
            })
    else:
        st.info(
            "Configure the patient in the sidebar, then click **Run prediction**. "
            "Three sample profiles are available; you can also build a custom patient."
        )


# ============================================================================
# Tab 2: DRIFT DETECTION
# ============================================================================
with tab_drift:
    st.subheader("Drift detection — overview")
    st.markdown(
        "CerviRisk-MM watches its own input distributions with three "
        "statistical tests. The full version (Docker / local clone) pulls "
        "live NCBI deposits and refreshes every minute. **This cloud demo "
        "shows a static demonstration** of how the detector responds to a "
        "calibrated post-vaccination scenario."
    )

    st.subheader("Statistical tests")
    drift_methods = pd.DataFrame([
        {"Method": "PSI",
         "Detects":   "Categorical shift (strain composition, ancestry)",
         "Threshold": "< 0.10 none · 0.10–0.20 minor · ≥ 0.20 significant"},
        {"Method": "KS 2-sample",
         "Detects":   "Continuous shift (age, PRS, smoking years)",
         "Threshold": "D ≥ 0.10 OR p < 0.05"},
        {"Method": "Chi-square",
         "Detects":   "Observed counts vs published prior",
         "Threshold": "p < 0.05"},
    ])
    st.dataframe(drift_methods, use_container_width=True, hide_index=True)

    st.subheader("Forward-time vaccination simulation")
    st.caption(
        "Calibrated to Drolet et al. *Lancet* 2019 — pooled meta-analysis of "
        "65 studies and 60 million person-years. HPV16 down 80%, HPV18 down "
        "83%, HPV31/33/45 cross-protection 55–65%, plateau at year 13."
    )
    drift_sim = pd.DataFrame([
        {"Time": "Baseline", "HPV16 share": "55.0%", "PSI": 0.000,
         "Severity": "none",        "Recommended": "—"},
        {"Time": "Year 5",   "HPV16 share": "50.7%", "PSI": 0.020,
         "Severity": "none",        "Recommended": "no action"},
        {"Time": "Year 10",  "HPV16 share": "42.1%", "PSI": 0.156,
         "Severity": "minor",       "Recommended": "monitor"},
        {"Time": "Year 15",  "HPV16 share": "35.9%", "PSI": 0.330,
         "Severity": "significant", "Recommended": "RETRAIN"},
        {"Time": "Year 20",  "HPV16 share": "31.2%", "PSI": 0.495,
         "Severity": "significant", "Recommended": "RETRAIN"},
    ])
    st.dataframe(drift_sim, use_container_width=True, hide_index=True)

    st.markdown(
        "The detector stays **silent during natural variation** (years 0–5), "
        "raises a **graduated warning** as drift accumulates (year 10 → "
        "minor), and crosses the **retrain threshold** at the realistic "
        "15–20 year horizon predicted by the Drolet meta-analysis."
    )

    st.subheader("Real-world finding")
    st.info(
        "When the same detector is applied to **live NCBI HPV deposits vs "
        "the de Sanjosé 2010 published prevalence prior**, it returns "
        "**PSI = 1.68** — significant drift. This is not a population "
        "shift; it is **research-deposition bias** (HPV16 is over-"
        "represented in NCBI sequencing studies). The detector correctly "
        "surfaces it as a data-quality signal."
    )

    baseline = load_baseline()
    if baseline:
        with st.expander("Saved drift baseline (training-time statistics)"):
            st.json(baseline)


# ============================================================================
# Tab 3: ABOUT
# ============================================================================
with tab_about:
    st.subheader("About this demo")
    st.markdown(
        "**CerviRisk-MM** is a research prototype multi-modal cervical "
        "cancer risk prediction pipeline. This demo runs the deployed model "
        "only — the full pipeline (FastAPI service, drift detection against "
        "live NCBI feed, 53 automated tests, Docker deployment, GitHub "
        "Actions CI) is available in the source repository."
    )

    st.subheader("Deployed model — performance")
    st.caption(f"Variant: `{DEPLOYED_METRICS['variant']}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DEV AUPRC",   f"{DEPLOYED_METRICS['dev']['auprc']}%")
    c2.metric("DEV AUROC",   f"{DEPLOYED_METRICS['dev']['auroc']}%")
    c3.metric("Sensitivity", f"{DEPLOYED_METRICS['dev']['sensitivity']}%")
    c4.metric("Specificity", f"{DEPLOYED_METRICS['dev']['specificity']}%")
    st.caption(
        f"Evaluated by leave-one-out cross-validation on "
        f"{DEPLOYED_METRICS['dev']['n']} development patients "
        f"({DEPLOYED_METRICS['dev']['positives']} biopsy-positive)."
    )

    with st.expander("Held-out TEST set (5-fold stratified subsampling)"):
        test = DEPLOYED_METRICS["test"]
        st.markdown(
            f"- **AUPRC** {test['auprc'][0]}% ± {test['auprc'][1]}%\n"
            f"- **AUROC** {test['auroc'][0]}% ± {test['auroc'][1]}%\n"
            f"- **Sensitivity** {test['sensitivity'][0]}% ± {test['sensitivity'][1]}%\n"
            f"- **Specificity** {test['specificity'][0]}% ± {test['specificity'][1]}%\n\n"
            f"n = {test['n']} held-out patients · {test['positives']} biopsy-positive · "
            f"wide standard deviations reflect ~2 positives per 5-fold chunk."
        )

    st.subheader("Honest limitations")
    st.markdown(
        "- **Small training cohort** (n = 858 from a single Venezuelan clinic). "
        "Geographic generalization is not validated.\n"
        "- **The deployed model is `triage + xgb`** — a referral decision-support "
        "tool that assumes prior screening tests exist (Hinselmann, Schiller, "
        "cytology). It is not a primary screening tool.\n"
        "- **The host PRS is `BIOLOGICALLY_INFORMED_SYNTHETIC`** — real GWAS "
        "biology, but per-individual genotypes are sampled from population "
        "allele frequencies, not from real VCFs.\n"
        "- **Augmentation did not improve over UCI-only features** on this "
        "cohort — reported honestly because the architecture is the deliverable, "
        "not the metric."
    )

    st.subheader("Source")
    st.markdown("- **GitHub:** https://github.com/nibrasissa/cervirisk-mm")
    st.markdown("- **License:** MIT")

st.divider()
st.caption(
    f"Generated at {datetime.utcnow().isoformat(timespec='seconds')} UTC · "
    "research prototype — not a medical device."
)

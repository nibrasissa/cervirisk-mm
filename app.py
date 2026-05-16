"""CerviRisk-MM — Streamlit Community Cloud deployment.

Self-contained: loads the trained model directly with joblib, AND fetches
live HPV strain composition from NCBI E-utilities to compute drift against
the published de Sanjose 2010 prior. No separate FastAPI process needed.

Live: https://cervirisk-mm.streamlit.app/
"""
from __future__ import annotations

import json
import math
import re
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
BRAND_LINE        = "rgba(128,128,128,0.18)"

TIER_COLORS = {
    "low":      BRAND_NAVY,
    "moderate": BRAND_AMBER,
    "high":     BRAND_MAGENTA,
}

DISCLAIMER = (
    "Research prototype, not a medical device. Predictions must not be "
    "used for clinical diagnosis or treatment decisions without appropriate "
    "regulatory approval and clinical validation."
)

# Deployed model — frozen v0.1 metrics
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

# Published prior — de Sanjose et al. Lancet Oncology 11(11):1048 (2010).
# Pooled analysis of HPV type distribution in invasive cervical cancer.
DE_SANJOSE_2010_PRIOR = {
    "HPV16":  0.55,
    "HPV18":  0.15,
    "HPV31":  0.05,
    "HPV33":  0.05,
    "HPV45":  0.04,
    "HPV52":  0.03,
    "HPV58":  0.03,
    "OTHER":  0.10,
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
# Live NCBI drift — inlined so no FastAPI process needed
# ---------------------------------------------------------------------------
HPV_TYPE_RE = re.compile(r"(?:type|HPV[-\s]?)(\d{1,3})", re.IGNORECASE)
KNOWN_HR_TYPES = {"HPV16", "HPV18", "HPV31", "HPV33", "HPV45", "HPV52", "HPV58"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_ncbi_strain_counts(n_records: int = 200) -> tuple[dict, dict]:
    """Pull recent HPV deposits from NCBI and aggregate by type."""
    from Bio import Entrez
    Entrez.email = "cervirisk-demo@streamlit.app"

    started = datetime.utcnow()

    h = Entrez.esearch(
        db="nucleotide",
        term="human papillomavirus[Organism]",
        retmax=n_records,
        sort="pub_date",
    )
    res = Entrez.read(h)
    h.close()
    ids = res.get("IdList", [])

    if not ids:
        return {}, {"started": started.isoformat(), "n_records": 0,
                     "n_typed": 0, "source": "NCBI nucleotide (live)"}

    h = Entrez.esummary(db="nucleotide", id=",".join(ids))
    summaries = Entrez.read(h)
    h.close()

    counts: dict[str, int] = {}
    n_typed = 0
    for s in summaries:
        title = str(s.get("Title", ""))
        m = HPV_TYPE_RE.search(title)
        if not m:
            continue
        n = int(m.group(1))
        if n > 200:
            continue
        key = f"HPV{n}" if f"HPV{n}" in KNOWN_HR_TYPES else "OTHER"
        counts[key] = counts.get(key, 0) + 1
        n_typed += 1

    meta = {
        "started":   started.isoformat(timespec="seconds") + "Z",
        "n_records": len(ids),
        "n_typed":   n_typed,
        "source":    "NCBI nucleotide via Entrez (live, on-demand)",
    }
    return counts, meta


def compute_psi(observed: dict, expected: dict, epsilon: float = 1e-6) -> float:
    """Population Stability Index."""
    keys = set(observed) | set(expected)
    total = sum(observed.values()) or 1
    psi = 0.0
    for k in keys:
        p_obs = max(observed.get(k, 0) / total, epsilon)
        p_exp = max(expected.get(k, 0),         epsilon)
        psi += (p_obs - p_exp) * math.log(p_obs / p_exp)
    return psi


def severity_for_psi(psi: float) -> tuple[str, str, str]:
    """Return (severity, recommendation, color)."""
    if psi < 0.10:
        return "none",        "No action required.",          BRAND_NAVY
    if psi < 0.20:
        return "minor",       "Monitor; investigate causes.", BRAND_AMBER
    return     "significant", "Retrain recommended.",         BRAND_MAGENTA


# ---------------------------------------------------------------------------
# Forecast helpers (forward projection of strain composition + predicted positives)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)   # 15-min cache, this is heavier than the snapshot call
def fetch_ncbi_strain_timeseries(
    days_back: int = 90,
    max_records: int = 1000,
) -> tuple[pd.DataFrame, dict]:
    """Pull HPV deposits from NCBI for the last `days_back` days, grouped by
    week. Returns weekly strain shares so each row sums to 1.

    Output DataFrame: index = ISO week start, one column per strain.
    """
    from Bio import Entrez
    from datetime import date, datetime, timedelta

    Entrez.email = "cervirisk-demo@streamlit.app"

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=days_back)
    started = datetime.utcnow()

    # esearch with date filter + history (cleaner for larger batches)
    h = Entrez.esearch(
        db="nucleotide",
        term="human papillomavirus[Organism]",
        mindate=start_date.strftime("%Y/%m/%d"),
        maxdate=today.strftime("%Y/%m/%d"),
        datetype="pdat",
        retmax=max_records,
        sort="pub_date",
        usehistory="y",
    )
    res = Entrez.read(h)
    h.close()
    ids = res.get("IdList", [])

    meta = {
        "started":     started.isoformat(timespec="seconds") + "Z",
        "days_back":   days_back,
        "date_window": f"{start_date} to {today}",
        "n_returned":  len(ids),
        "n_typed":     0,
        "source":      "NCBI nucleotide via Entrez (live, weekly bucketed)",
    }

    if not ids:
        return pd.DataFrame(), meta

    h = Entrez.esummary(
        db="nucleotide",
        WebEnv=res["WebEnv"],
        query_key=res["QueryKey"],
        retmax=max_records,
    )
    summaries = Entrez.read(h)
    h.close()

    rows = []
    for s in summaries:
        title = str(s.get("Title", ""))
        m = HPV_TYPE_RE.search(title)
        if not m:
            continue
        n = int(m.group(1))
        if n > 200:
            continue
        strain = f"HPV{n}" if f"HPV{n}" in KNOWN_HR_TYPES else "OTHER"

        # Pub date can be "YYYY/MM/DD", "YYYY/MM", "YYYY", or "YYYY MonthName DD"
        pub_date_str = str(s.get("PubDate") or s.get("CreateDate") or "").strip()
        parsed = None
        for fmt in ("%Y/%m/%d", "%Y/%m", "%Y", "%Y %b %d", "%Y %b"):
            try:
                parsed = datetime.strptime(pub_date_str, fmt).date()
                break
            except ValueError:
                continue
        if parsed is None:
            continue
        rows.append({"date": parsed, "strain": strain})

    meta["n_typed"] = len(rows)

    if not rows:
        return pd.DataFrame(), meta

    df = pd.DataFrame(rows)
    # Bucket by ISO week so the chart has roughly weekly granularity
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time
    counts = df.groupby(["week", "strain"]).size().unstack(fill_value=0)

    # Ensure every known strain has a column for a stable legend
    for s in list(KNOWN_HR_TYPES) + ["OTHER"]:
        if s not in counts.columns:
            counts[s] = 0
    # Stable column order
    counts = counts[sorted(KNOWN_HR_TYPES) + ["OTHER"]]

    totals = counts.sum(axis=1).replace(0, 1)
    shares = counts.div(totals, axis=0)

    return shares, meta


# ---------------------------------------------------------------------------
# Pipeline helpers (prediction)
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
    """Subtle tier indicator with a colored left accent."""
    color = TIER_COLORS.get(tier, BRAND_MUTED)
    st.markdown(f"""
        <div style="border-left:6px solid {color};
                     padding:14px 18px;border-radius:4px;
                     background:rgba(128,128,128,0.06);
                     font-family:sans-serif;">
            <div style="font-size:0.8em;letter-spacing:0.12em;
                         opacity:0.75;color:var(--text-color);">
                PREDICTED RISK
            </div>
            <div style="font-size:2.1em;font-weight:600;line-height:1.1;
                         color:{color};margin-top:2px;">
                {prob*100:.1f}%
            </div>
            <div style="font-size:0.85em;letter-spacing:0.08em;margin-top:4px;
                         color:var(--text-color);opacity:0.85;">
                Tier: <b>{tier.lower()}</b>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_psi_card(psi: float, severity: str, recommendation: str, color: str) -> None:
    """Subtle PSI result with accent strip + clean metric block."""
    st.markdown(f"""
        <div style="border-left:6px solid {color};
                     padding:14px 20px;border-radius:4px;
                     background:rgba(128,128,128,0.06);
                     font-family:sans-serif;margin:8px 0 16px 0;">
            <div style="font-size:0.8em;letter-spacing:0.12em;
                         opacity:0.75;color:var(--text-color);">
                PSI &nbsp;·&nbsp; OBSERVED VS DE SANJOSE 2010 PRIOR
            </div>
            <div style="font-size:2.4em;font-weight:600;line-height:1.1;
                         color:{color};margin-top:2px;">
                {psi:.3f}
            </div>
            <div style="font-size:0.9em;margin-top:6px;
                         color:var(--text-color);opacity:0.9;">
                Severity: <b>{severity}</b>. {recommendation}
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
    st.caption("Multi-modal cervical cancer risk prediction. Research prototype.")
    st.divider()

    model, model_path = load_model()
    if model is None:
        st.error("Model artifact not found in the repo.")
        st.caption(
            "If you're running locally, train the model with `.\\run.ps1 fit` "
            "or `.\\run.ps1 train`."
        )
        st.stop()
    st.success("Model loaded")
    st.caption(f"`{model_path}`")

    st.divider()
    profile_choice = st.radio(
        "Pick a patient profile",
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
        with st.expander("Multi-modal context (genetics and virology)"):
            strain_pick = st.selectbox(
                "Assigned HPV strain",
                ["(none - no HPV detected)", "HPV16", "HPV18", "HPV31",
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
    "Multi-modal cervical cancer risk prediction. Research prototype, "
    "not a medical device."
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
            st.markdown(f"  **{age_val} yo** patient. Ancestry: **{pop}**.")
            cl = []
            if patient.get("Smokes"):
                cl.append(f"smoker ({patient.get('Smokes (years)', '?')}y)")
            screening_pos = sum(int(bool(patient.get(k, 0))) for k in
                                  ("Hinselmann", "Schiller", "Citology"))
            if screening_pos > 0:
                cl.append(f"{screening_pos}/3 screening tests positive")
            if cl:
                st.markdown(f"  Clinical: {', '.join(cl)}.")
            if patient.get("host_prs") is not None:
                st.markdown(f"  Host genetics: PRS {patient['host_prs']:.2f}.")
            strain = patient.get("assigned_hpv_strain")
            if strain:
                carc = patient.get("strain_carcinogenicity")
                line = f"  Viral: {strain}"
                if carc is not None:
                    line += f", carcinogenicity {carc}"
                st.markdown(line + ".")

        with col_risk:
            render_tier_badge(tier, proba)
            rec = {
                "low":      "Routine screening, next cycle.",
                "moderate": "Expedited follow-up; repeat cytology in 6 months.",
                "high":     "Refer for diagnostic biopsy.",
            }[tier]
            st.markdown(f"<div style='margin-top:10px;font-size:0.9em;"
                        f"opacity:0.85;'>{rec}</div>",
                        unsafe_allow_html=True)

        st.divider()
        st.subheader("Feature contributions")
        st.caption(
            "Top features pushing this prediction toward higher (magenta) or "
            "lower (navy) risk. Computed via XGBoost built-in TreeSHAP."
        )
        contributors = compute_shap(model, df, top_k=10)
        if contributors and "error" not in contributors[0]:
            max_abs = max(abs(c["contribution"]) for c in contributors)
            for i, c in enumerate(contributors, 1):
                color = BRAND_MAGENTA_BAR if c["direction"] == "up" else BRAND_NAVY_BAR
                arrow = "+" if c["direction"] == "up" else "−"
                width = abs(c["contribution"]) / max_abs * 100
                raw_val = resolve_raw_value(c["name"], patient)
                st.markdown(f"""
                    <div style="display:grid;
                                 grid-template-columns:28px 220px 1fr 110px 90px;
                                 align-items:center;font-family:sans-serif;
                                 font-size:0.9em;padding:5px 0;
                                 border-bottom:1px solid {BRAND_LINE};">
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
                                     font-weight:600;font-variant-numeric:tabular-nums;">
                            {arrow}{abs(c['contribution']):.3f}
                        </div>
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
            "Configure the patient in the sidebar, then click Run prediction. "
            "Three sample profiles are available; you can also build a custom patient."
        )


# ============================================================================
# Tab 2: DRIFT DETECTION (live NCBI)
# ============================================================================
with tab_drift:
    st.subheader("Drift detection")
    st.markdown(
        "Pulls recent HPV sequence deposits directly from the NCBI Entrez "
        "API and computes drift against the published de Sanjose 2010 "
        "cervical-cancer prevalence prior using PSI (Population Stability "
        "Index). Refreshes automatically; results are cached for 5 minutes "
        "to be polite to NCBI."
    )

    # Sample-size control + manual refresh, both compact.
    col_n, col_refresh = st.columns([3, 1])
    with col_n:
        n_records = st.selectbox(
            "Records to fetch",
            options=[100, 200, 500],
            index=1,
            help="NCBI fetch size. Larger is more accurate but slower.",
            label_visibility="collapsed",
        )
    with col_refresh:
        if st.button("Refresh now", use_container_width=True,
                      help="Force a fresh NCBI fetch, bypassing the 5-minute cache."):
            fetch_live_ncbi_strain_counts.clear()
            st.rerun()

    # Automatic fetch on every page render (cached for 5 minutes).
    counts, meta = {}, {}
    try:
        with st.spinner("Fetching live HPV deposits from NCBI..."):
            counts, meta = fetch_live_ncbi_strain_counts(n_records=n_records)
    except Exception as e:
        st.error(
            f"NCBI fetch failed: `{type(e).__name__}: {e}`. "
            "Possible causes: NCBI rate limit, network issue, or Biopython "
            "not installed. The page will retry on the next refresh."
        )

    if counts:
        psi = compute_psi(counts, DE_SANJOSE_2010_PRIOR)
        severity, recommendation, color = severity_for_psi(psi)

        render_psi_card(psi, severity, recommendation, color)

        with st.expander("Fetch metadata"):
            st.json(meta)

        st.markdown("##### Interpretation")
        if severity == "significant":
            st.markdown(
                "The deposit composition deviates significantly from the "
                "published cervical-cancer prior. Possible reasons:\n\n"
                "- Real epidemiological shift, for example post-vaccination "
                "strain replacement.\n"
                "- Research-deposition bias. NCBI is dominated by sequencing "
                "studies, not population epidemiology. HPV16 is over-"
                "represented relative to its true prevalence. This is the "
                "most likely interpretation today.\n"
                "- A new emerging strain reaching wider sequencing attention."
            )
        elif severity == "minor":
            st.markdown(
                "Mild drift detected. Worth monitoring; not yet at the "
                "retrain threshold."
            )
        else:
            st.markdown(
                "No meaningful drift. The current NCBI deposit composition "
                "is consistent with the de Sanjose 2010 prior."
            )
    else:
        st.info(
            "No NCBI records typed yet. The fetch may still be retrying; "
            "use Refresh now to try again."
        )

    st.divider()
    st.subheader("Live strain composition over time")
    st.markdown(
        "Each line is the weekly share of one HPV strain in NCBI deposits "
        "over the last 90 days. All data is fetched live from NCBI Entrez "
        "at the time of viewing and grouped by publication week."
    )

    col_days, col_max = st.columns([3, 1])
    with col_days:
        days_back = st.selectbox(
            "Window",
            options=[30, 60, 90, 180],
            index=2,
            help="How many days of NCBI deposits to include.",
            label_visibility="collapsed",
        )
    with col_max:
        if st.button("Refresh series",
                      help="Force a fresh fetch of the time series",
                      use_container_width=True):
            fetch_ncbi_strain_timeseries.clear()
            st.rerun()

    try:
        with st.spinner(f"Fetching the last {days_back} days from NCBI..."):
            series_df, series_meta = fetch_ncbi_strain_timeseries(
                days_back=days_back,
                max_records=1000,
            )
    except Exception as e:
        st.error(
            f"NCBI time-series fetch failed: `{type(e).__name__}: {e}`. "
            "This can happen with rate limits or network hiccups; retry in a moment."
        )
        series_df, series_meta = pd.DataFrame(), {}

    if isinstance(series_df, pd.DataFrame) and not series_df.empty:
        st.line_chart(series_df, height=350)
        st.caption(
            "Weekly share, one line per HPV type. Weeks with very few "
            "typed records may look spiky. Use the window selector above "
            "to widen the time range."
        )
        with st.expander("Time-series fetch metadata"):
            st.json(series_meta)
    else:
        st.info(
            "No NCBI records were returned for the selected window. "
            "Try widening the window or use Refresh series."
        )

    with st.expander("Methods reference: three statistical tests"):
        st.dataframe(pd.DataFrame([
            {"Method": "PSI",
             "Detects":   "Categorical shift (strain composition, ancestry)",
             "Threshold": "< 0.10 none, 0.10–0.20 minor, ≥ 0.20 significant"},
            {"Method": "KS 2-sample",
             "Detects":   "Continuous shift (age, PRS, smoking years)",
             "Threshold": "D ≥ 0.10 or p < 0.05"},
            {"Method": "Chi-square",
             "Detects":   "Observed counts vs published prior",
             "Threshold": "p < 0.05"},
        ]), use_container_width=True, hide_index=True)
        st.caption(
            "PSI is the categorical drift metric used above. KS and chi-square "
            "are used by the full pipeline on continuous and count features."
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
        "This is the live demo of CerviRisk-MM, a multi-modal cervical "
        "cancer risk prediction pipeline. The demo runs the deployed model "
        "and the live NCBI drift check. The full pipeline, including the "
        "FastAPI service, auto-refreshing dashboard, 53 automated tests, "
        "Docker deployment, and GitHub Actions CI, is available in the "
        "source repository."
    )

    st.subheader("Deployed model: performance")
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
            f"- AUPRC {test['auprc'][0]}% ± {test['auprc'][1]}%\n"
            f"- AUROC {test['auroc'][0]}% ± {test['auroc'][1]}%\n"
            f"- Sensitivity {test['sensitivity'][0]}% ± {test['sensitivity'][1]}%\n"
            f"- Specificity {test['specificity'][0]}% ± {test['specificity'][1]}%\n\n"
            f"n = {test['n']} held-out patients, {test['positives']} biopsy-positive. "
            f"Wide standard deviations reflect approximately 2 positives per 5-fold chunk."
        )

    st.subheader("Limitations")
    st.markdown(
        "- Small training cohort (n = 858 from a single Venezuelan clinic). "
        "Geographic generalization is not validated.\n"
        "- The deployed model is `triage + xgb`, a referral decision-support "
        "tool that assumes prior screening tests exist (Hinselmann, Schiller, "
        "cytology). It is not a primary screening tool.\n"
        "- The host PRS is `BIOLOGICALLY_INFORMED_SYNTHETIC`: real GWAS "
        "biology, but per-individual genotypes are sampled from population "
        "allele frequencies, not from real VCFs.\n"
        "- Augmentation did not improve over UCI-only features on this cohort."
    )

    st.subheader("Source")
    st.markdown("- GitHub: https://github.com/nibrasissa/cervirisk-mm")

st.divider()
st.caption(
    f"Generated at {datetime.utcnow().isoformat(timespec='seconds')} UTC. "
    "Research prototype, not a medical device."
)

"""CerviRisk-MM v0.1 — nested evaluation with hyperparameter tuning.

Updated with Sophisticated Imputation (IterativeImputer).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# Add IterativeImputer and its required experimental enable flag
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.features.build_patient import (
    SCREENING_TEST_COLUMNS,
    UCI_FEATURE_COLUMNS,
    assemble_patients,
)
from src.model.metrics import (
    aggregate_fold_metrics,
    compute_all_metrics,
    f1_threshold,
    format_aggregated_metrics_block,
    format_metrics_block,
    youden_threshold,
)
from src.storage.paths import MODELS_DIR

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)


AUGMENTED_NUMERIC = ["strain_carcinogenicity", "host_prs"]
AUGMENTED_CATEGORICAL = ["assigned_hpv_strain", "matched_super_pop"]
TRIAGE_NUMERIC = SCREENING_TEST_COLUMNS  
MODELS = ("logreg", "logreg_cal", "rf", "balanced_rf", "gbm", "xgb")
MODES = ("uci_only", "augmented", "triage")
OUTER_SEED = 42
INNER_SEED = 42
THRESHOLD_STRATEGIES = ("default_0_5", "youden", "f1")

TUNE_N_ITER = 20
TUNE_CV_FOLDS = 3
TUNE_SCORING = "average_precision"

SEARCH_SPACES: dict[str, dict] = {
    "logreg": {
        "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
    },
    "logreg_cal": {
        "estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0],
    },
    "rf": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", "log2"],
    },
    "balanced_rf": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", "log2"],
    },
    "gbm": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 5, 10],
    },
    "xgb": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
}

def _build_xy(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = df["biopsy_outcome"].fillna(0).astype(int).values
    uci_cols = [c for c in UCI_FEATURE_COLUMNS if c in df.columns]
    if mode == "uci_only":
        X = df[uci_cols].copy()
    elif mode == "augmented":
        X = df[uci_cols + AUGMENTED_NUMERIC + AUGMENTED_CATEGORICAL].copy()
    elif mode == "triage":
        triage_cols = [c for c in TRIAGE_NUMERIC if c in df.columns]
        X = df[uci_cols + triage_cols].copy()
    else:
        raise ValueError(f"unknown mode: {mode}")
    return X, y

def _make_preprocessor(mode: str) -> ColumnTransformer:
    numeric_aug = AUGMENTED_NUMERIC if mode == "augmented" else []
    numeric_triage = TRIAGE_NUMERIC if mode == "triage" else []
    numeric_all = list(UCI_FEATURE_COLUMNS) + numeric_aug + numeric_triage
    categorical = AUGMENTED_CATEGORICAL if mode == "augmented" else []

    # REPLACED SimpleImputer with IterativeImputer for sophisticated estimation
    numeric_pipe = Pipeline([
        ("impute", IterativeImputer(random_state=INNER_SEED, max_iter=10)),
        ("scale", StandardScaler()),
    ])
    
    transformers: list = [("num", numeric_pipe, numeric_all)]
    if categorical:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def _make_pipeline(
    mode: str,
    model_kind: str,
    seed: int,
    tuned_params: dict | None = None,
) -> Pipeline:
    preprocessor = _make_preprocessor(mode)

    if model_kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif model_kind == "logreg_cal":
        base = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    elif model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    elif model_kind == "balanced_rf":
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier
        except ImportError as e:
            raise ImportError(
                "balanced_rf requires imbalanced-learn: pip install imbalanced-learn"
            ) from e
        clf = BalancedRandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1,
            sampling_strategy="auto", replacement=False, bootstrap=True,
        )
    elif model_kind == "gbm":
        clf = GradientBoostingClassifier(random_state=seed)
    elif model_kind == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgb requires xgboost: pip install xgboost") from e
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed, n_jobs=-1, eval_metric="logloss", tree_method="hist",
        )
    else:
        raise ValueError(model_kind)

    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    if tuned_params:
        prefixed = {f"clf__{k}": v for k, v in tuned_params.items()}
        pipe.set_params(**prefixed)

    return pipe

def _fit_with_balanced_weights(pipe: Pipeline, X, y) -> Pipeline:
    clf = pipe.named_steps["clf"]
    needs_sample_weight = isinstance(clf, GradientBoostingClassifier) or (
        clf.__class__.__name__ == "XGBClassifier"
    )
    if needs_sample_weight:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        pipe.fit(X, y, clf__sample_weight=weights)
    else:
        pipe.fit(X, y)
    return pipe

def assert_no_leakage(train_index, test_index, X) -> None:
    train_set = set(map(int, train_index))
    test_set = set(map(int, test_index))
    overlap = train_set & test_set
    if overlap:
        raise AssertionError(
            f"DATA LEAKAGE: {len(overlap)} indices in both train & test."
        )

def tune_variant(X_dev, y_dev, mode, kind, n_iter=TUNE_N_ITER, seed=INNER_SEED) -> dict:
    space = SEARCH_SPACES.get(kind, {})
    if not space:
        logger.info("    no search space for %s — skipping tuning", kind)
        return {}

    pipe = _make_pipeline(mode, kind, seed)
    pipe_space = {f"clf__{k}": v for k, v in space.items()}
    inner_cv = StratifiedKFold(n_splits=TUNE_CV_FOLDS, shuffle=True, random_state=seed)

    fit_kwargs: dict = {}
    if kind in ("gbm", "xgb"):
        weights = compute_sample_weight(class_weight="balanced", y=y_dev)
        fit_kwargs["clf__sample_weight"] = weights

    search = RandomizedSearchCV(
        pipe,
        param_distributions=pipe_space,
        n_iter=n_iter,
        cv=inner_cv,
        scoring=TUNE_SCORING,
        n_jobs=-1,
        random_state=seed,
        refit=False,
        verbose=0,
    )
    start = time.time()
    search.fit(X_dev, y_dev, **fit_kwargs)
    elapsed = time.time() - start

    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    logger.info("    tuned %s+%s in %.0fs  best AUPRC=%.3f  params=%s",
                mode, kind, elapsed, float(search.best_score_), best)
    return best

def _pick_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "default_0_5": 0.5,
        "youden": youden_threshold(y_true, y_proba),
        "f1": f1_threshold(y_true, y_proba),
    }

def _metrics_at_each_threshold(y_true, y_proba, thresholds) -> dict[str, dict]:
    return {
        name: compute_all_metrics(y_true, y_proba, threshold=t)
        for name, t in thresholds.items()
    }

def _loocv_oof_predictions(X, y, mode, kind, tuned_params, label="DEV") -> np.ndarray:
    loo = LeaveOneOut()
    n = len(y)
    proba_oof = np.zeros(n)
    start = time.time()
    log_every = max(50, n // 8)

    for i, (train_idx, test_idx) in enumerate(loo.split(X)):
        assert_no_leakage(train_idx, test_idx, X)
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y[train_idx]
        pipe = _make_pipeline(mode, kind, INNER_SEED, tuned_params=tuned_params)
        _fit_with_balanced_weights(pipe, X_tr, y_tr)
        proba_oof[test_idx[0]] = pipe.predict_proba(X_te)[0, 1]
        if (i + 1) % log_every == 0:
            elapsed = time.time() - start
            eta = (n - i - 1) / ((i + 1) / elapsed)
            logger.info("    [%s LOOCV] %s+%s  fold %d/%d  ETA=%.0fs",
                        label, mode, kind, i + 1, n, eta)

    elapsed = time.time() - start
    logger.info("    [%s LOOCV] %s+%s  done in %.1fs", label, mode, kind, elapsed)
    return proba_oof

def _test_kfold_stability(X_dev, y_dev, X_test, y_test, mode, kind,
                          dev_thresholds, tuned_params, n_folds=5) -> dict:
    pipe = _make_pipeline(mode, kind, INNER_SEED, tuned_params=tuned_params)
    _fit_with_balanced_weights(pipe, X_dev, y_dev)
    proba_test = pipe.predict_proba(X_test)[:, 1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=OUTER_SEED)
    per_threshold_folds: dict[str, list[dict]] = {n: [] for n in THRESHOLD_STRATEGIES}

    for _, fold_idx in skf.split(X_test, y_test):
        y_fold = y_test[fold_idx]
        p_fold = proba_test[fold_idx]
        if len(np.unique(y_fold)) < 2:
            continue
        for name, t in dev_thresholds.items():
            per_threshold_folds[name].append(
                compute_all_metrics(y_fold, p_fold, threshold=t)
            )

    return {name: aggregate_fold_metrics(folds)
            for name, folds in per_threshold_folds.items()}

def _eval_split(X, y, mode, kind, tuned_params) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=OUTER_SEED
    )
    assert_no_leakage(X_tr.index.to_numpy(), X_te.index.to_numpy(), X)
    pipe = _make_pipeline(mode, kind, INNER_SEED, tuned_params=tuned_params)
    _fit_with_balanced_weights(pipe, X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    thresholds = _pick_thresholds(y_te, proba)
    test = _metrics_at_each_threshold(y_te, proba, thresholds)
    return {"mode": mode, "model": kind, "eval": "split",
            "dev": None, "test": test, "thresholds": thresholds,
            "tuned_params": tuned_params}

def _eval_loocv(X, y, mode, kind, tuned_params) -> dict:
    proba_oof = _loocv_oof_predictions(X, y, mode, kind, tuned_params, label="ALL")
    thresholds = _pick_thresholds(y, proba_oof)
    dev = _metrics_at_each_threshold(y, proba_oof, thresholds)
    return {"mode": mode, "model": kind, "eval": "loocv",
            "dev": dev, "test": None, "thresholds": thresholds,
            "tuned_params": tuned_params}

def _eval_nested(X, y, mode, kind, tuned_params) -> dict:
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=OUTER_SEED
    )
    assert_no_leakage(X_dev.index.to_numpy(), X_test.index.to_numpy(), X)

    proba_oof_dev = _loocv_oof_predictions(X_dev, y_dev, mode, kind, tuned_params, label="DEV")
    thresholds = _pick_thresholds(y_dev, proba_oof_dev)
    dev = _metrics_at_each_threshold(y_dev, proba_oof_dev, thresholds)
    test = _test_kfold_stability(
        X_dev, y_dev, X_test, y_test, mode, kind, thresholds, tuned_params
    )
    return {"mode": mode, "model": kind, "eval": "nested",
            "dev": dev, "test": test, "thresholds": thresholds,
            "tuned_params": tuned_params}

EVAL_FUNCS = {"split": _eval_split, "loocv": _eval_loocv, "nested": _eval_nested}

def train_all(eval_mode="nested", models=MODELS, tune=False) -> tuple[Pipeline, list[dict]]:
    logger.info("Assembling patients…")
    df = assemble_patients()
    eval_fn = EVAL_FUNCS[eval_mode]

    results: list[dict] = []
    final_models: dict[tuple[str, str], Pipeline] = {}
    tuned_params_by_variant: dict[str, dict] = {}

    for mode in MODES:
        X, y = _build_xy(df, mode=mode)
        logger.info("Mode=%s, X=%s, prevalence=%.3f", mode, X.shape, y.mean())

        if eval_mode == "nested":
            X_for_tune, _, y_for_tune, _ = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=OUTER_SEED
            )
        else:
            X_for_tune, y_for_tune = X, y

        for kind in models:
            logger.info("  → variant %s+%s [%s%s]", mode, kind, eval_mode,
                        " +tune" if tune else "")
            tuned_params: dict = {}
            if tune:
                try:
                    tuned_params = tune_variant(X_for_tune, y_for_tune, mode, kind)
                except ImportError as e:
                    logger.warning("    skipping %s (tuning): %s", kind, e)
                    continue
                tuned_params_by_variant[f"{mode}+{kind}"] = tuned_params

            try:
                result = eval_fn(X, y, mode, kind, tuned_params)
            except ImportError as e:
                logger.warning("    skipping %s (eval): %s", kind, e)
                continue
            results.append(result)
            _log_variant(result)

            full_pipe = _make_pipeline(mode, kind, INNER_SEED, tuned_params=tuned_params)
            _fit_with_balanced_weights(full_pipe, X, y)
            final_models[(mode, kind)] = full_pipe

    if tune and tuned_params_by_variant:
        (MODELS_DIR / "tuned_params.json").write_text(
            json.dumps(tuned_params_by_variant, indent=2, default=str)
        )
        logger.info("Saved tuned hyperparameters to %s",
                    MODELS_DIR / "tuned_params.json")

    best = _pick_best_variant(results)
    logger.info("Best variant: mode=%s, kind=%s (by DEV AUPRC where available)",
                best["mode"], best["model"])
    return final_models[(best["mode"], best["model"])], results

def _is_aggregated(metrics_dict: dict) -> bool:
    return any(isinstance(k, str) and k.endswith("_mean") for k in metrics_dict.keys())

def _metric_value(metrics_dict: dict, name: str, default=None):
    if not metrics_dict:
        return default
    if _is_aggregated(metrics_dict):
        return metrics_dict.get(f"{name}_mean", default)
    return metrics_dict.get(name, default)

def _format_block(metrics_dict: dict) -> str:
    if _is_aggregated(metrics_dict):
        return format_aggregated_metrics_block(metrics_dict)
    return format_metrics_block(metrics_dict)

def _log_variant(result: dict) -> None:
    dev = result.get("dev") or {}
    test = result.get("test") or {}
    dev_y = dev.get("youden", {}) if isinstance(dev, dict) else {}
    test_y = test.get("youden", {}) if isinstance(test, dict) else {}
    parts = []
    if dev_y:
        parts.append(
            f"DEV  AUPRC={_metric_value(dev_y, 'auprc', float('nan')):.3f}  "
            f"AUROC={_metric_value(dev_y, 'auroc', float('nan')):.3f}  "
            f"Sens={_metric_value(dev_y, 'sensitivity', float('nan')):.3f}  "
            f"Spec={_metric_value(dev_y, 'specificity', float('nan')):.3f}  "
            f"F1={_metric_value(dev_y, 'f1', float('nan')):.3f}"
        )
    if test_y:
        if _is_aggregated(test_y):
            parts.append(
                f"TEST AUPRC={test_y.get('auprc_mean', float('nan')):.3f}±"
                f"{test_y.get('auprc_std', float('nan')):.3f}  "
                f"Sens={test_y.get('sensitivity_mean', float('nan')):.3f}±"
                f"{test_y.get('sensitivity_std', float('nan')):.3f}  "
                f"Spec={test_y.get('specificity_mean', float('nan')):.3f}±"
                f"{test_y.get('specificity_std', float('nan')):.3f}"
            )
        else:
            parts.append(
                f"TEST AUPRC={test_y.get('auprc', float('nan')):.3f}  "
                f"Sens={test_y.get('sensitivity', float('nan')):.3f}  "
                f"Spec={test_y.get('specificity', float('nan')):.3f}"
            )
    if parts:
        logger.info("    %s", " | ".join(parts))

def _pick_best_variant(results: list[dict]) -> dict:
    def score(r):
        dev = r.get("dev") or {}
        if isinstance(dev, dict) and dev:
            v = dev.get("youden", {}).get("auprc")
            if v is not None:
                return v
        test = r.get("test") or {}
        if isinstance(test, dict) and test:
            youden = test.get("youden", {})
            return youden.get("auprc_mean") or youden.get("auprc") or 0
        return 0
    return max(results, key=score)

def save_artifacts(pipe: Pipeline, results: list[dict], eval_mode: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "cervirisk_mm_v0.1.pkl")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    _write_comparison_md(results, eval_mode)
    _write_full_metrics_md(results, eval_mode)
    logger.info("Saved artifacts to %s", MODELS_DIR)

def _to_pct(v):
    return None if v is None else round(100 * v, 1)

def _write_comparison_md(results: list[dict], eval_mode: str) -> None:
    rows = []
    for r in results:
        dev = (r.get("dev") or {}).get("youden", {}) if isinstance(r.get("dev"), dict) else {}
        test = (r.get("test") or {}).get("youden", {}) if isinstance(r.get("test"), dict) else {}
        rows.append({
            "mode": r["mode"], "model": r["model"],
            "dev_AUPRC_%": _to_pct(_metric_value(dev, "auprc")),
            "dev_AUROC_%": _to_pct(_metric_value(dev, "auroc")),
            "dev_Sens_%": _to_pct(_metric_value(dev, "sensitivity")),
            "dev_Spec_%": _to_pct(_metric_value(dev, "specificity")),
            "dev_F1_%": _to_pct(_metric_value(dev, "f1")),
            "dev_MCC": _metric_value(dev, "mcc"),
            "test_AUPRC_%": _to_pct(_metric_value(test, "auprc")),
            "test_AUPRC_std_%": _to_pct(test.get("auprc_std")),
            "test_AUROC_%": _to_pct(_metric_value(test, "auroc")),
            "test_AUROC_std_%": _to_pct(test.get("auroc_std")),
            "test_Sens_%": _to_pct(_metric_value(test, "sensitivity")),
            "test_Spec_%": _to_pct(_metric_value(test, "specificity")),
            "test_F1_%": _to_pct(_metric_value(test, "f1")),
        })
    df = pd.DataFrame(rows).round(3)
    (MODELS_DIR / "comparison.md").write_text(
        f"# CerviRisk-MM v0.1 — model comparison ({eval_mode}, Youden threshold)\n\n"
        f"All rate metrics shown as percentages (%). MCC is a correlation in [-1, +1].\n\n"
        f"- **DEV** (`dev_*` columns): LOOCV on the 80% development set (n=686, 44 positives).\n"
        f"- **TEST** (`test_*` columns): 5-fold stability evaluation on the 20% held-out test set "
        f"(n=172, 11 positives). Test data never participates in any model fit.\n\n"
        f"```\n{df.to_string(index=False)}\n```\n\n"
        f"Sensitivity, specificity, F1, MCC are computed at the Youden-optimal "
        f"threshold derived from DEV out-of-fold predictions and applied to TEST.\n"
    )

def _write_full_metrics_md(results: list[dict], eval_mode: str) -> None:
    if not results:
        return
    best = _pick_best_variant(results)
    lines = [
        f"# CerviRisk-MM v0.1 — full metrics for best variant",
        f"",
        f"Mode: **{best['mode']}** Model: **{best['model']}** Eval: **{best['eval']}**",
        f"",
    ]
    tp = best.get("tuned_params") or {}
    if tp:
        lines.append(f"Tuned hyperparameters: `{tp}`")
        lines.append("")
    for level in ("dev", "test"):
        block = best.get(level)
        if not isinstance(block, dict) or not block:
            continue
        level_label = (
            "DEV — LOOCV on the 80% development set"
            if level == "dev" else
            "TEST — 5-fold stability evaluation on the 20% held-out set"
        )
        lines.append(f"## {level_label}")
        lines.append("")
        lines.append("Rate metrics in % (1 decimal). Others as labeled.")
        lines.append("")
        for thr_name in THRESHOLD_STRATEGIES:
            tm = block.get(thr_name, {})
            if not tm:
                continue
            lines.append(f"### Threshold: {thr_name}")
            lines.append("```")
            lines.append(_format_block(tm))
            lines.append("```")
            lines.append("")
    (MODELS_DIR / "metrics_full.md").write_text("\n".join(lines))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CerviRisk-MM v0.1")
    parser.add_argument("--eval", choices=tuple(EVAL_FUNCS), default="nested")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help=f"Subset of: {', '.join(MODELS)}")
    parser.add_argument("--tune", action="store_true",
                        help="Run RandomizedSearchCV per variant on DEV before evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    pipe, results = train_all(eval_mode=args.eval, models=tuple(args.models), tune=args.tune)
    save_artifacts(pipe, results, args.eval)

    print()
    print("=" * 86)
    print(f"CerviRisk-MM v0.1 — training complete (eval={args.eval}{' +tune' if args.tune else ''})")
    print("=" * 86)
    print("Rate metrics shown as percentages (%).")
    print(f"DEV  = LOOCV on the 80% development set (~686 patients, ~44 positives).")
    print(f"TEST = 5-fold stability eval on the 20% held-out set (~172 patients, ~11 positives).")
    print()

    rows = []
    for r in results:
        dev = (r.get("dev") or {}).get("youden", {}) if isinstance(r.get("dev"), dict) else {}
        test = (r.get("test") or {}).get("youden", {}) if isinstance(r.get("test"), dict) else {}
        rows.append({
            "mode": r["mode"], "model": r["model"],
            "DEV_AUPRC%": _to_pct(_metric_value(dev, "auprc")),
            "DEV_Sens%": _to_pct(_metric_value(dev, "sensitivity")),
            "DEV_Spec%": _to_pct(_metric_value(dev, "specificity")),
            "DEV_F1%": _to_pct(_metric_value(dev, "f1")),
            "TEST_AUPRC%": _to_pct(_metric_value(test, "auprc")),
            "TEST_Sens%": _to_pct(_metric_value(test, "sensitivity")),
            "TEST_Spec%": _to_pct(_metric_value(test, "specificity")),
            "TEST_F1%": _to_pct(_metric_value(test, "f1")),
        })
    df = pd.DataFrame(rows).round(1)
    print(df.to_string(index=False))
    print()

    best = _pick_best_variant(results)
    print("=" * 86)
    print(f"BEST VARIANT: {best['mode']} + {best['model']}")
    if best.get("tuned_params"):
        print(f"Tuned hyperparameters: {best['tuned_params']}")
    print("=" * 86)
    for level in ("dev", "test"):
        block = best.get(level)
        if not isinstance(block, dict) or not block:
            continue
        for thr_name in THRESHOLD_STRATEGIES:
            tm = block.get(thr_name, {})
            if not tm:
                continue
            level_label = (
                "DEV (LOOCV on 80% set)" if level == "dev"
                else "TEST (5-fold stability on 20% held-out)"
            )
            print(f"\n[{level_label} | threshold={thr_name}]")
            print(_format_block(tm))
    print()
    print("Artifacts written to:", MODELS_DIR)

if __name__ == "__main__":
    main()
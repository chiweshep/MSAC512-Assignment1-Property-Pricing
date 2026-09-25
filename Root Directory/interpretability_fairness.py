#!/usr/bin/env python3
"""Section E -- residual analysis, SHAP global/local interpretability,
location-vs-condition-vs-size decomposition, fairness evidence table.

IMPORTANT: this script reads the SAME train/val/test split that
model_development.py wrote to /content/models/split_indices.csv. It
does NOT create its own split. That keeps Section E's residual analysis
and SHAP explanations computed on exactly the test rows Section D's
comparison table was computed on -- if you change this, cross-check your
Section D and E numbers, because they will silently diverge.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


def categorise_feature(col: str, suburb_col: str) -> str:
    base = col.split("__")[-1] if "__" in col else col
    if base == suburb_col or base.startswith(f"{suburb_col}_") or base.startswith(suburb_col):
        return "location"
    if base.startswith("flag_") or base.startswith("img_") or base == "construction_stage_ordinal":
        return "condition_finish"
    if base in ("bedrooms", "bathrooms", "land_area_m2", "floor_area_m2"):
        return "size_bedroom"
    return "other"


# ---------- Residual analysis ----------

def residual_analysis(y_true, y_pred, suburb, land_area, outdir, alpha=0.05):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=15)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted price (USD)")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs fitted")
    axes[1].scatter(land_area, residuals, alpha=0.5, s=15)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Land area (m²)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs size")
    fig.tight_layout()
    fig.savefig(outdir / "residuals_vs_fitted_and_size.png", dpi=150)
    plt.close(fig)

    df = pd.DataFrame({"suburb": suburb.values, "residual": residuals})
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("suburb")["residual"].median().sort_values().index
    sns.boxplot(data=df, x="suburb", y="residual", order=order, ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals by suburb")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outdir / "residuals_by_suburb.png", dpi=150)
    plt.close(fig)

    rows = []
    for s in df["suburb"].unique():
        vals = df.loc[df["suburb"] == s, "residual"]
        if len(vals) < 3:
            continue
        t_stat, p = stats.ttest_1samp(vals, popmean=0)
        rows.append({"suburb": s, "n": len(vals),
                     "mean_residual_usd": vals.mean(),
                     "t_stat": t_stat, "p_value_raw": p})
    tab = pd.DataFrame(rows)
    tab["p_value_bonferroni"] = (tab["p_value_raw"] * len(tab)).clip(upper=1.0)
    tab["systematic_mispricing_flag"] = tab["p_value_bonferroni"] < alpha
    tab = tab.sort_values("mean_residual_usd")
    print("\n--- Residual analysis by suburb ---")
    print(tab.round(2).to_string(index=False))
    n_flag = int(tab["systematic_mispricing_flag"].sum())
    print(f"\n{n_flag} suburb(s) show statistically significant non-zero mean residual "
          f"after Bonferroni correction (alpha={alpha}). Positive mean residual = "
          f"model UNDER-predicts in that suburb.")
    return tab


# ---------- SHAP ----------

def compute_shap_values(model, X_background, X_explain):
    """Returns (shap_values, feature_names, X_transformed, explainer_type,
    base_value). base_value is explainer.expected_value -- NOT mean(log price);
    using the latter would make the waterfall not sum to the prediction."""
    pre = model.named_steps["preprocess"]
    est = model.named_steps["model"]
    X_bg_t = pre.transform(X_background)
    X_ex_t = pre.transform(X_explain)
    feature_names = list(pre.get_feature_names_out())

    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_ex_t)
        base = float(np.array(explainer.expected_value).ravel()[0])
        return sv, feature_names, X_ex_t, "TreeExplainer", base
    except Exception as e:
        print(f"TreeExplainer failed ({e}); using model-agnostic explainer "
              f"(slower -- one row at a time).")
        predict_fn = lambda X: est.predict(X)
        n_bg = min(50, X_bg_t.shape[0])
        bg = shap.kmeans(X_bg_t, n_bg)
        explainer = shap.KernelExplainer(predict_fn, bg)
        n_exp = min(30, X_ex_t.shape[0])
        print(f"  KernelExplainer on {n_exp} rows ...")
        sv = explainer.shap_values(X_ex_t[:n_exp], nsamples=100)
        base = float(np.array(explainer.expected_value).ravel()[0])
        return sv, feature_names, X_ex_t[:n_exp], "KernelExplainer", base


def global_importance(shap_values, feature_names, outdir, top_n=20):
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * min(top_n, len(importance)))))
    top = importance.head(top_n)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    ax.set_xlabel("mean |SHAP value| (impact on log price)")
    ax.set_title(f"Global SHAP importance (top {top_n})")
    fig.tight_layout()
    fig.savefig(outdir / "shap_global_importance.png", dpi=150)
    plt.close(fig)
    return importance


def permutation_importance_check(model, X_test, y_test_log, n_repeats=10):
    result = permutation_importance(
        model, X_test, y_test_log, n_repeats=n_repeats,
        random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)


def build_display_values(feature_names, explain_df, numeric_cols, suburb_col):
    stripped = [f.split("__")[-1] if "__" in f else f for f in feature_names]
    if not all(s in numeric_cols or s.startswith(suburb_col) for s in stripped):
        return None
    display = np.zeros((len(explain_df), len(feature_names)))
    for j, s in enumerate(stripped):
        if s in numeric_cols:
            display[:, j] = explain_df[s].values
        else:
            suffix = s[len(suburb_col) + 1:] if s.startswith(f"{suburb_col}_") else None
            if suffix is not None:
                display[:, j] = (explain_df[suburb_col].values == suffix).astype(float)
    return display


def local_explanations(shap_values, feature_names, X_transformed, explain_df,
                        base_value, outdir, numeric_cols=None, suburb_col="suburb"):
    display = None
    if numeric_cols is not None:
        display = build_display_values(feature_names, explain_df, numeric_cols, suburb_col)
        if display is None:
            print("  (using transformed feature values in the waterfall labels)")

    for i, (_, row) in enumerate(explain_df.iterrows()):
        data_for_plot = display[i] if display is not None else X_transformed[i]
        exp = shap.Explanation(
            values=shap_values[i], base_values=base_value,
            data=data_for_plot, feature_names=list(feature_names))
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp, show=False, max_display=15)
        plt.title(f"Local explanation: ref={row.get('ref', i)} "
                  f"(actual=${row.get('price_usd', float('nan')):,.0f})")
        plt.tight_layout()
        out = outdir / f"shap_waterfall_{row.get('ref', i)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def decompose_by_category(shap_values, feature_names, suburb_col):
    categories = [categorise_feature(f, suburb_col) for f in feature_names]
    abs_shap = np.abs(shap_values)
    totals = pd.Series(abs_shap.sum(axis=0)).groupby(pd.Series(categories)).sum()
    shares = (totals / totals.sum() * 100).sort_values(ascending=False)
    out = shares.reset_index()
    out.columns = ["category", "pct_of_total_abs_shap"]
    print("\n--- Location vs condition/finish vs size/bedroom decomposition ---")
    print(out.round(1).to_string(index=False))
    print("\nUse these percentages directly in Section F and Section 6 -- do not "
          "re-derive a different split there.")
    return out


def fairness_evidence_table(shap_values, feature_names, explain_df, suburb_col):
    idx = [i for i, f in enumerate(feature_names)
           if (f.split("__")[-1] if "__" in f else f).startswith(suburb_col)]
    if not idx:
        print("No suburb one-hot columns in SHAP names -- skipping fairness table.")
        return pd.DataFrame()
    loc_shap = shap_values[:, idx].sum(axis=1)
    tab = pd.DataFrame({"suburb": explain_df[suburb_col].values,
                          "location_shap_contribution": loc_shap})
    summary = tab.groupby("suburb")["location_shap_contribution"].agg(["mean", "count"])
    summary = summary.sort_values("mean", ascending=False)
    print("\n--- Fairness evidence: mean location-only SHAP by suburb ---")
    print(summary.round(4).to_string())
    print("""
DISCUSS THIS YOURSELF in the Zimbabwean context the brief asks for:
  1. Does the ranking track the historical high-density/low-density
     (formerly Group-Areas-era) geography of Harare?
  2. If RBZ or a Board asked you to justify the location loading, which
     specific artefact from this assignment would you table?
  3. Is there a LEGITIMATE channel (flood risk, infrastructure, distance
     to CBD) this dataset lets you distinguish from a pure price-level
     effect? If not, say so.
""")
    return summary


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="/content/model_ready_features.csv")
    p.add_argument("--model-dir", default="/content/models")
    p.add_argument("--model-name", default="xgboost",
                    help="One of GLM_baseline, random_forest, xgboost, "
                         "elasticnet_interactions.")
    p.add_argument("--outdir", default="/content/interpret")
    p.add_argument("--price-col",     default="price_usd")
    p.add_argument("--suburb-col",    default="suburb")
    p.add_argument("--bedrooms-col",  default="bedrooms")
    p.add_argument("--bathrooms-col", default="bathrooms")
    p.add_argument("--land-area-col", default="land_area_m2")
    p.add_argument("--flag-prefix",   default="flag_")
    p.add_argument("--image-prefix",  default="img_")
    p.add_argument("--explain-refs", nargs="*", default=None)
    p.add_argument("--n-shap-sample", type=int, default=200)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.model_dir)

    df = pd.read_csv(args.features)
    splits = pd.read_csv(mdir / "split_indices.csv")
    df = df.merge(splits, on="ref", how="left")
    cols = json.load(open(mdir / "feature_cols.json"))

    model_path = mdir / f"{args.model_name}.joblib"
    model = joblib.load(model_path)
    print(f"Loaded {model_path} ({args.model_name}); {len(df)} total listings")

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"Using saved split: test n={len(test_df)}, train n={len(train_df)}")

    numeric_cols = cols["numeric"]
    categorical_cols = cols["categorical"]
    feature_cols = numeric_cols + categorical_cols

    test_df = test_df.dropna(subset=feature_cols + [args.price_col]).reset_index(drop=True)

    # predict
    if args.model_name == "GLM_baseline":
        # statsmodels GLM: predict via formula-style DataFrame + smearing
        glm_cols = ["suburb", "bedrooms", "bathrooms", "land_area_m2"]
        pred_df = test_df.dropna(subset=glm_cols).copy()
        log_pred = model.predict(pred_df)
        smear = float(np.mean(np.exp(model.resid)))
        price_pred = np.exp(log_pred) * smear
        test_pred_df = pred_df
    else:
        log_pred = model.predict(test_df[feature_cols])
        price_pred = np.exp(log_pred)
        test_pred_df = test_df

    # --- residuals ---
    suburb_stats = residual_analysis(
        test_pred_df[args.price_col].values, price_pred,
        test_pred_df[args.suburb_col], test_pred_df[args.land_area_col],
        outdir)
    suburb_stats.to_csv(outdir / "residuals_by_suburb.csv", index=False)

    # --- SHAP only for the tree/linear ML models, not the statsmodels GLM ---
    if args.model_name == "GLM_baseline":
        print("\nGLM baseline: skipping SHAP (its coefficients in "
              "models/glm_coefficients.csv are the interpretability output).")
    else:
        shap_sample = test_pred_df.sample(
            n=min(args.n_shap_sample, len(test_pred_df)),
            random_state=RANDOM_STATE).reset_index(drop=True)
        background = train_df.sample(
            n=min(100, len(train_df)), random_state=RANDOM_STATE)

        print(f"\nComputing SHAP for {len(shap_sample)} test rows "
              f"(background n={len(background)})...")
        sv, feat_names, Xt, explainer_used, base_value = compute_shap_values(
            model, background[feature_cols], shap_sample[feature_cols])
        print(f"Used {explainer_used}; base_value = {base_value:.4f} (log price)")

        imp = global_importance(sv, feat_names, outdir)
        imp.to_csv(outdir / "shap_global_importance.csv", index=False)
        print("\n--- Top 10 global SHAP features ---")
        print(imp.head(10).round(4).to_string(index=False))

        y_log = np.log(shap_sample[args.price_col].clip(lower=1))
        perm = permutation_importance_check(model, shap_sample[feature_cols], y_log)
        perm.to_csv(outdir / "permutation_importance.csv", index=False)
        print("\n--- Top 10 permutation-importance features (SHAP cross-check) ---")
        print(perm.head(10).round(4).to_string(index=False))

        # local explanations: pick over/under-predicted
        shap_sample = shap_sample.iloc[:len(sv)].reset_index(drop=True)
        log_pred_sample = np.array([feat for feat in [None]])  # placeholder
        # recompute predictions on the (possibly truncated) shap_sample
        if args.model_name == "GLM_baseline":
            pred_short = price_pred[:len(shap_sample)]
        else:
            pred_short = np.exp(model.predict(shap_sample[feature_cols]))
        shap_sample["_pred"] = pred_short
        shap_sample["_resid"] = shap_sample[args.price_col] - shap_sample["_pred"]

        if args.explain_refs:
            idx = shap_sample.index[shap_sample["ref"].isin(args.explain_refs)].tolist()
        else:
            idx = [shap_sample["_resid"].idxmax(), shap_sample["_resid"].idxmin()]
            print(f"\nAuto-selected: biggest under-prediction ref="
                  f"{shap_sample.loc[idx[0], 'ref']}; biggest over-prediction "
                  f"ref={shap_sample.loc[idx[1], 'ref']}")

        # Xt index positions map to the same rows because shap_sample
        # was reset above before SHAP was computed on X_explain
        print("\nGenerating SHAP waterfall plots ...")
        local_explanations(sv[idx], feat_names, Xt[idx],
                            shap_sample.loc[idx], base_value, outdir,
                            numeric_cols=numeric_cols, suburb_col=args.suburb_col)

        decomp = decompose_by_category(sv, feat_names, args.suburb_col)
        decomp.to_csv(outdir / "shap_category_decomposition.csv", index=False)

        fair = fairness_evidence_table(sv, feat_names, shap_sample, args.suburb_col)
        if not fair.empty:
            fair.to_csv(outdir / "shap_location_contribution_by_suburb.csv")

    print(f"\nAll Section E outputs written to {outdir}/")


if __name__ == "__main__":
    main()

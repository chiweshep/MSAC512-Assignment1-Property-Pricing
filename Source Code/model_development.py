#!/usr/bin/env python3
"""Section D -- GLM baseline + 3 ML models + region-stratified split +
nested tuning + conformal prediction intervals + comparison table.

STRATIFICATION STRATEGY
-----------------------
Your scraped sample has ~397 listings across ~106 suburbs; the median
suburb has 3 listings. Stratifying on suburb is impossible at that
granularity (sklearn's train_test_split requires >=2 per group per
split, so ~100 suburbs would fail). We therefore stratify on the
COARSER 'address_region' field (Harare North/West/East/South/High
Density/Ruwa/Chitungwiza/Norton/Mashonaland East/Mashonaland West),
which preserves spatial representation across splits. State this
choice and its consequence in Section 5.3: the split prevents a whole
region from being absent from any split, but individual suburbs may
still be train-only or test-only, and the model uses suburb as a
feature -- that is fine for the tree models (unseen levels get ignored
at prediction time) and handled explicitly for the GLM (see
predict_glm below).
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
GLM_COLS = ["suburb", "bedrooms", "bathrooms", "land_area_m2"]


def stratified_split(df, strat_col, train_frac=0.6, val_frac=0.2,
                      random_state=RANDOM_STATE):
    """Stratified split on strat_col. Falls back to non-stratified if any
    group has <2 members (sklearn refuses stratification in that case).
    Clamps fractional sizes so train_size is always in (0, 1)."""
    df = df.reset_index(drop=True).copy()
    counts = df[strat_col].value_counts()
    small_groups = counts[counts < 2].index
    use_strat = len(small_groups) == 0
    if not use_strat:
        print(f"WARNING: {len(small_groups)} group(s) in '{strat_col}' have "
              f"<2 listings; falling back to non-stratified split. Groups: "
              f"{sorted(small_groups.tolist())[:8]}{'...' if len(small_groups) > 8 else ''}")

    n = len(df)
    n_train = max(1, int(round(train_frac * n)))
    n_val   = max(1, int(round(val_frac * n)))

    if use_strat:
        train_df, temp_df = train_test_split(
            df, train_size=n_train,
            stratify=df[strat_col], random_state=random_state)
        val_frac_of_temp = n_val / len(temp_df)
        # guard: must remain in (0, 1) for train_test_split
        val_frac_of_temp = min(max(val_frac_of_temp, 0.05), 0.95)
        try:
            val_df, test_df = train_test_split(
                temp_df, train_size=val_frac_of_temp,
                stratify=temp_df[strat_col], random_state=random_state)
        except ValueError as e:
            print(f"  second-split stratification failed ({e}); "
                  f"falling back to non-stratified for val/test.")
            val_df, test_df = train_test_split(
                temp_df, train_size=val_frac_of_temp,
                random_state=random_state)
    else:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(n)
        train_df = df.iloc[idx[:n_train]]
        val_df   = df.iloc[idx[n_train:n_train + n_val]]
        test_df  = df.iloc[idx[n_train + n_val:]]

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def fit_glm_baseline(train_df, price_col, suburb_col, bedrooms_col,
                      bathrooms_col, land_area_col):
    d = train_df.dropna(subset=[price_col, suburb_col, bedrooms_col,
                                  bathrooms_col, land_area_col]).copy()
    d["log_price"] = np.log(d[price_col].clip(lower=1))
    formula = (f"log_price ~ C({suburb_col}) + {bedrooms_col} + "
               f"{bathrooms_col} + {land_area_col}")
    model = smf.ols(formula, data=d).fit()
    coef_table = pd.DataFrame({
        "coefficient": model.params,
        "pct_price_effect": (np.exp(model.params) - 1) * 100,
        "p_value": model.pvalues,
        "ci_low_pct":  (np.exp(model.conf_int()[0]) - 1) * 100,
        "ci_high_pct": (np.exp(model.conf_int()[1]) - 1) * 100,
    })
    print("\n--- 5.1 GLM baseline (OLS on log price) ---")
    print(model.summary())
    print(coef_table.drop(index="Intercept")[
        ["pct_price_effect", "p_value"]].round(2).to_string())
    print(f"\nAdjusted R^2: {model.rsquared_adj:.3f}")
    return model, coef_table


def predict_glm(model, df, train_suburbs=None):
    """Predict on ORIGINAL price scale with Duan's smearing correction.
    Handles unseen suburb levels by mapping them to the most frequent
    training suburb (bias is small; effect is documented in Section 5.3)."""
    df = df.dropna(subset=GLM_COLS).copy()
    if train_suburbs is not None and len(df):
        unseen = ~df["suburb"].isin(train_suburbs)
        if unseen.any():
            fallback = sorted(train_suburbs)[0]
            print(f"  [predict_glm] {int(unseen.sum())} row(s) with unseen "
                  f"suburb mapped to '{fallback}'.")
            df.loc[unseen, "suburb"] = fallback
    log_pred = model.predict(df)
    smearing = float(np.mean(np.exp(model.resid)))
    return np.exp(log_pred) * smearing


def build_preprocessor(numeric_cols, categorical_cols, interaction_degree=None):
    num_steps = [("scale", StandardScaler())]
    if interaction_degree:
        num_steps.append(("poly", PolynomialFeatures(
            degree=interaction_degree, interaction_only=True, include_bias=False)))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


MODEL_SPECS = {
    "random_forest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
        "param_distributions": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "uses_interactions": False,
    },
    "xgboost": {
        "estimator": XGBRegressor(random_state=RANDOM_STATE,
                                    objective="reg:squarederror"),
        "param_distributions": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.1, 1.0, 5.0],
        },
        "uses_interactions": False,
    },
    "elasticnet_interactions": {
        "estimator": ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        "param_distributions": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "uses_interactions": True,
    },
}


def fit_ml_model(name, spec, train_df, numeric_cols, categorical_cols,
                  target, n_iter=25, inner_folds=5):
    preprocessor = build_preprocessor(
        numeric_cols, categorical_cols,
        interaction_degree=2 if spec["uses_interactions"] else None)
    pipe = Pipeline([("preprocess", preprocessor), ("model", spec["estimator"])])
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE)
    n_candidates = min(n_iter, int(np.prod(
        [len(v) for v in spec["param_distributions"].values()])))
    search = RandomizedSearchCV(
        pipe, spec["param_distributions"], n_iter=n_candidates, cv=inner_cv,
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE,
        n_jobs=-1, refit=True)
    X_train = train_df[numeric_cols + categorical_cols]
    print(f"\nTuning {name}: {inner_folds}-fold inner CV, {n_candidates} "
          f"candidates (TRAIN split only)...")
    search.fit(X_train, target)
    print(f"  best inner-CV RMSE (log price): {-search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def conformal_interval_halfwidth(y_true, y_pred, alpha=0.1):
    residuals = np.abs(y_true - y_pred)
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level))


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE_pct": mape, "R2": r2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="/content/model_ready_features.csv")
    p.add_argument("--outdir", default="/content/models")
    p.add_argument("--price-col",     default="price_usd")
    p.add_argument("--suburb-col",    default="suburb")
    p.add_argument("--region-col",    default="address_region",
                    help="Coarser location variable used for stratification. "
                         "Falls back to suburb if not present.")
    p.add_argument("--bedrooms-col",  default="bedrooms")
    p.add_argument("--bathrooms-col", default="bathrooms")
    p.add_argument("--land-area-col", default="land_area_m2")
    p.add_argument("--flag-prefix",   default="flag_")
    p.add_argument("--image-prefix",  default="img_")
    p.add_argument("--alpha",  type=float, default=0.1)
    p.add_argument("--n-iter", type=int,   default=25)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} listings, {len(df.columns)} columns")
    df = df.dropna(subset=[args.price_col, args.suburb_col]).copy()

    # choose stratification variable
    if args.region_col in df.columns and df[args.region_col].nunique() >= 4:
        strat_col = args.region_col
        print(f"\nStratifying on '{strat_col}' "
              f"({df[strat_col].nunique()} groups; region sizes: "
              f"{df[strat_col].value_counts().to_dict()})")
    else:
        strat_col = args.suburb_col
        print(f"\n'{args.region_col}' unavailable or too coarse -- "
              f"stratifying on '{strat_col}' instead.")
    train_df, val_df, test_df = stratified_split(df, strat_col)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, "
          f"test={len(test_df)}")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name}: {part[args.suburb_col].nunique()} suburbs, "
              f"{part[strat_col].nunique()} regions")

    splits = pd.concat([
        train_df[["ref"]].assign(split="train"),
        val_df[["ref"]].assign(split="val"),
        test_df[["ref"]].assign(split="test"),
    ], ignore_index=True)
    splits.to_csv(outdir / "split_indices.csv", index=False)

    glm_model, glm_coefs = fit_glm_baseline(
        train_df, args.price_col, args.suburb_col,
        args.bedrooms_col, args.bathrooms_col, args.land_area_col)
    glm_coefs.to_csv(outdir / "glm_coefficients.csv")

    glm_train_suburbs = set(
        train_df.dropna(subset=GLM_COLS)[args.suburb_col].unique())
    print(f"\nGLM was fitted on {len(glm_train_suburbs)} suburb levels.")

    glm_val_eval  = val_df.dropna(subset=GLM_COLS).reset_index(drop=True)
    glm_test_eval = test_df.dropna(subset=GLM_COLS).reset_index(drop=True)
    glm_val_pred  = predict_glm(glm_model, glm_val_eval,
                                  train_suburbs=glm_train_suburbs)
    glm_test_pred = predict_glm(glm_model, glm_test_eval,
                                  train_suburbs=glm_train_suburbs)

    glm_halfwidth = conformal_interval_halfwidth(
        glm_val_eval[args.price_col].values, glm_val_pred, args.alpha)

    all_metrics = {"GLM_baseline": compute_metrics(
        glm_test_eval[args.price_col].values, glm_test_pred)}
    all_metrics["GLM_baseline"]["interval_halfwidth_usd"] = glm_halfwidth
    all_metrics["GLM_baseline"]["interval_coverage_target_pct"] = (1 - args.alpha) * 100

    numeric_cols = [args.bedrooms_col, args.bathrooms_col, args.land_area_col]
    numeric_cols += [c for c in df.columns if c.startswith(args.flag_prefix)
                      and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols += [c for c in df.columns if c.startswith(args.image_prefix)
                      and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in dict.fromkeys(numeric_cols) if c in df.columns]
    categorical_cols = [args.suburb_col]
    print(f"\nFull feature set: {len(numeric_cols)} numeric, "
          f"{len(categorical_cols)} categorical")

    with open(outdir / "feature_cols.json", "w") as f:
        json.dump({"numeric": numeric_cols, "categorical": categorical_cols,
                    "price_col": args.price_col, "alpha": args.alpha}, f, indent=2)

    fitted_models = {"GLM_baseline": glm_model}
    best_params_all = {}
    halfwidths = {"GLM_baseline": glm_halfwidth}

    for name, spec in MODEL_SPECS.items():
        train_sub = train_df.dropna(subset=numeric_cols).reset_index(drop=True)
        y_sub = np.log(train_sub[args.price_col].clip(lower=1))
        fitted, best = fit_ml_model(name, spec, train_sub, numeric_cols,
                                     categorical_cols, y_sub.values,
                                     n_iter=args.n_iter)
        fitted_models[name] = fitted
        best_params_all[name] = best

        val_sub  = val_df.dropna(subset=numeric_cols).reset_index(drop=True)
        test_sub = test_df.dropna(subset=numeric_cols).reset_index(drop=True)
        val_pred  = np.exp(fitted.predict(val_sub[numeric_cols + categorical_cols]))
        test_pred = np.exp(fitted.predict(test_sub[numeric_cols + categorical_cols]))
        hw = conformal_interval_halfwidth(
            val_sub[args.price_col].values, val_pred, args.alpha)
        halfwidths[name] = hw
        m = compute_metrics(test_sub[args.price_col].values, test_pred)
        m["interval_halfwidth_usd"] = hw
        m["interval_coverage_target_pct"] = (1 - args.alpha) * 100
        all_metrics[name] = m

    with open(outdir / "best_params.json", "w") as f:
        json.dump({k: {kk: (vv if not isinstance(vv, np.generic) else vv.item())
                        for kk, vv in v.items()}
                    for k, v in best_params_all.items()}, f, indent=2)
    with open(outdir / "conformal_halfwidths.json", "w") as f:
        json.dump(halfwidths, f, indent=2)

    comparison = pd.DataFrame(all_metrics).T
    comparison = comparison[["RMSE", "MAE", "MAPE_pct", "R2",
                              "interval_halfwidth_usd",
                              "interval_coverage_target_pct"]]
    comparison.to_csv(outdir / "model_comparison.csv")
    print("\n" + "=" * 70)
    print("5.5 MODEL COMPARISON (test set, original price scale)")
    print("=" * 70)
    print(comparison.round(2).to_string())
    print(f"\nWritten to {outdir / 'model_comparison.csv'}")
    print("""
SELECTION IS YOUR JUDGEMENT CALL. Weigh RMSE/MAE/MAPE against
interpretability AND interval width -- the lowest RMSE alone is not
sufficient justification, per the brief.""")

    for name, model in fitted_models.items():
        joblib.dump(model, outdir / f"{name}.joblib")
    print(f"\nSaved all fitted models to {outdir}/*.joblib")


if __name__ == "__main__":
    main()

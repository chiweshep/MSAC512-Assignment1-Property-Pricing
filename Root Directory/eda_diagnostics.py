#!/usr/bin/env python3
"""
eda_diagnostics.py -- Section C (EDA + statistical diagnostics).
Reads /content/model_ready_features.csv; writes figures + tables to /content/eda/.
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm

sns.set_theme(style="whitegrid")


def analyse_price_distribution(df, price_col, outdir):
    price = df[price_col].dropna()
    log_price = np.log(price.clip(lower=1))
    skew_raw, skew_log = stats.skew(price), stats.skew(log_price)
    kurt_raw, kurt_log = stats.kurtosis(price), stats.kurtosis(log_price)
    shapiro_raw = stats.shapiro(price.sample(min(len(price), 4999), random_state=42))
    shapiro_log = stats.shapiro(log_price.sample(min(len(log_price), 4999), random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(price, kde=True, ax=axes[0])
    axes[0].set_title(f"Price (USD)\nskew={skew_raw:.2f}, Shapiro p={shapiro_raw.pvalue:.4f}")
    sns.histplot(log_price, kde=True, ax=axes[1])
    axes[1].set_title(f"log(Price)\nskew={skew_log:.2f}, Shapiro p={shapiro_log.pvalue:.4f}")
    fig.tight_layout()
    fig.savefig(outdir / "price_distribution.png", dpi=150)
    plt.close(fig)

    rec = ("log-price (or Gamma/Tweedie GLM with log link on raw price)"
           if abs(skew_raw) > 1
           else "raw price with OLS is defensible, but check residuals")
    result = {"n": len(price), "skewness_raw": skew_raw, "skewness_log": skew_log,
              "kurtosis_excess_raw": kurt_raw, "kurtosis_excess_log": kurt_log,
              "shapiro_p_raw": shapiro_raw.pvalue, "shapiro_p_log": shapiro_log.pvalue,
              "recommendation": rec}
    print("\n--- Price distribution ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def suburb_summary_table(df, price_col, suburb_col, land_area_col, bedrooms_col,
                          required_suburbs, min_listings=5):
    counts = df.groupby(suburb_col).size()
    coverage = df.groupby(suburb_col)[land_area_col].apply(lambda s: s.notna().mean())
    df = df.copy()
    df["price_per_m2"] = df[price_col] / df[land_area_col]
    df["price_per_bedroom"] = df[price_col] / df[bedrooms_col].replace(0, np.nan)
    rows = []
    for suburb, n in counts.items():
        if n < min_listings:
            continue
        sub = df[df[suburb_col] == suburb]
        cov = coverage.get(suburb, 0)
        metric = "price_per_m2" if cov >= 0.6 else "price_per_bedroom"
        rows.append({"suburb": suburb, "n_listings": int(n),
                      "land_area_coverage": round(cov, 2),
                      "metric_used": metric,
                      "median_value": sub[metric].median(),
                      "median_price_usd": sub[price_col].median()})
    table = (pd.DataFrame(rows).sort_values("median_value", ascending=False)
             .reset_index(drop=True))
    table["rank"] = table.index + 1
    missing = [s for s in required_suburbs if s not in table["suburb"].values]
    print(f"\n--- Suburb summary ({len(table)} suburbs with >={min_listings} listings) ---")
    print(table.to_string(index=False))
    if missing:
        print(f"\nWARNING: required suburb(s) not in table at min_listings={min_listings}: "
              f"{missing}.")
    return table


def dunn_posthoc(df, value_col, group_col):
    groups = sorted(df[group_col].unique())
    n_pairs = len(groups) * (len(groups) - 1) // 2
    pmat = pd.DataFrame(np.ones((len(groups), len(groups))), index=groups, columns=groups)
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            x = df.loc[df[group_col] == g1, value_col]
            y = df.loc[df[group_col] == g2, value_col]
            if len(x) < 2 or len(y) < 2:
                continue
            _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            p_adj = min(p * n_pairs, 1.0)
            pmat.loc[g1, g2] = p_adj; pmat.loc[g2, g1] = p_adj
    return pmat


def suburb_effect_test(df, price_col, suburb_col, land_area_col, bedrooms_col,
                        min_listings=5):
    counts = df[suburb_col].value_counts()
    valid = counts[counts >= min_listings].index
    d = df[df[suburb_col].isin(valid)].dropna(
        subset=[price_col, suburb_col, land_area_col, bedrooms_col]).copy()
    d["log_price"] = np.log(d[price_col].clip(lower=1))
    model = smf.ols(f"log_price ~ C({suburb_col}) + {land_area_col} + {bedrooms_col}",
                     data=d).fit()
    aov = anova_lm(model, typ=2)
    ss_suburb = aov.loc[f"C({suburb_col})", "sum_sq"]
    eta_sq = ss_suburb / aov["sum_sq"].sum()
    d["residual"] = model.resid
    groups = [g["residual"].values for _, g in d.groupby(suburb_col)]
    kw_stat, kw_p = stats.kruskal(*groups)
    print("\n--- Suburb effect on log(price), controlling for size + bedrooms ---")
    print(aov)
    print(f"\nEta-squared (suburb): {eta_sq:.3f} "
          f"({'small' if eta_sq < 0.06 else 'medium' if eta_sq < 0.14 else 'large'})")
    print(f"Kruskal-Wallis on OLS residuals by suburb: H={kw_stat:.2f}, p={kw_p:.4g}")
    posthoc = dunn_posthoc(d, "residual", suburb_col)
    print(f"Dunn's post-hoc (Bonferroni): "
          f"{(posthoc < 0.05).sum().sum() // 2} of {posthoc.size // 2} pairs "
          f"significant at alpha=0.05")
    return {"anova_table": aov, "eta_squared_suburb": eta_sq,
            "kruskal_wallis_stat": kw_stat, "kruskal_wallis_p": kw_p,
            "posthoc_pvalues": posthoc}


def check_multicollinearity(df, flag_cols, corr_threshold=0.7,
                              vif_threshold=5.0, outdir=None):
    X = df[flag_cols].astype(float).fillna(0)
    X = X[X.columns[X.std() > 0]]
    corr = X.corr()
    if outdir:
        fig, ax = plt.subplots(figsize=(max(8, len(X.columns) * 0.4),
                                          max(6, len(X.columns) * 0.35)))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_title("Condition-flag correlation matrix")
        fig.tight_layout()
        fig.savefig(outdir / "flag_correlation_matrix.png", dpi=150)
        plt.close(fig)
    high_pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                  for i in range(len(corr.columns))
                  for j in range(i + 1, len(corr.columns))
                  if abs(corr.iloc[i, j]) >= corr_threshold]
    Xc = sm.add_constant(X)
    vif = pd.DataFrame({
        "feature": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
    })
    vif = vif[vif["feature"] != "const"].sort_values("VIF", ascending=False)
    print("\n--- Multicollinearity check ---")
    print(f"Highly correlated pairs (|r| >= {corr_threshold}):")
    for a, b, r in high_pairs:
        print(f"  {a} <-> {b}: r={r:.2f}")
    print(f"\nVIF (>{vif_threshold} concerning):")
    print(vif.to_string(index=False))
    flagged = set(vif[vif["VIF"] > vif_threshold]["feature"])
    print(f"\nCandidates for dropping/combining: "
          f"{sorted(flagged) if flagged else 'none'}")
    return {"correlation_matrix": corr, "high_corr_pairs": high_pairs,
            "vif_table": vif, "vif_flagged": flagged}


def plot_condition_vs_price_by_suburb(df, price_col, suburb_col, condition_col,
                                        outdir, top_n_suburbs=8):
    top = df[suburb_col].value_counts().head(top_n_suburbs).index
    d = df[df[suburb_col].isin(top)].copy()
    d["log_price"] = np.log(d[price_col].clip(lower=1))
    is_binary = d[condition_col].dropna().isin([0, 1, True, False]).all()
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_binary:
        d[condition_col] = d[condition_col].astype(int)
        sns.boxplot(data=d, x=suburb_col, y="log_price",
                    hue=condition_col, ax=ax)
        ax.legend(title=condition_col)
    else:
        sns.scatterplot(data=d, x=condition_col, y="log_price",
                        hue=suburb_col, ax=ax)
    ax.set_title(f"log(price) vs {condition_col}, by suburb")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out = outdir / f"condition_vs_price_{condition_col}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path,
                    default=Path("/content/model_ready_features.csv"))
    p.add_argument("--outdir", type=Path, default=Path("/content/eda"))
    p.add_argument("--price-col",     default="price_usd")
    p.add_argument("--suburb-col",    default="suburb")
    p.add_argument("--land-area-col", default="land_area_m2")
    p.add_argument("--bedrooms-col",  default="bedrooms")
    p.add_argument("--condition-col", default="flag_kitchen_fitted")
    p.add_argument("--flag-prefix",   default="flag_")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found -- run merge_features.py first.")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} listings, {len(df.columns)} columns")

    warnings.filterwarnings("ignore", category=FutureWarning)

    analyse_price_distribution(df, args.price_col, args.outdir)
    suburb_summary_table(df, args.price_col, args.suburb_col,
                          args.land_area_col, args.bedrooms_col,
                          required_suburbs=["Madokero", "Mabvazuva"])
    suburb_effect_test(df, args.price_col, args.suburb_col,
                        args.land_area_col, args.bedrooms_col)

    flag_cols = [c for c in df.columns
                 if c.startswith(args.flag_prefix)
                 and df[c].dtype in (bool, int, float)]
    if len(flag_cols) >= 2:
        check_multicollinearity(df, flag_cols, outdir=args.outdir)
    else:
        print(f"\nSkipping VIF check -- found {len(flag_cols)} columns "
              f"matching prefix '{args.flag_prefix}' (need >=2).")

    if args.condition_col in df.columns:
        plot_condition_vs_price_by_suburb(df, args.price_col, args.suburb_col,
                                            args.condition_col, args.outdir)
    else:
        print(f"\nSkipping condition-vs-price plot -- "
              f"'{args.condition_col}' not in columns.")

    print(f"\nAll figures written to {args.outdir}/")


if __name__ == "__main__":
    main()

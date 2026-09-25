#!/usr/bin/env python3
"""Section F -- score House A (Madokero) and House B (Mabvazuva) through
the chosen final model, with prediction intervals, extrapolation
diagnostics, and a two-path decomposition of the A-B gap.

Reuses the artefacts from model_development.py so the case-study numbers
are consistent with Sections D and E. Fills any feature the House A/B
brief does not specify with the training median and logs which ones, so
the assumption is visible rather than silent. Reports extrapolation on
THREE axes: kNN distance, feature-value prevalence in the training
data, and the median price of the k nearest training listings -- so you
can compare the model's point estimate to what comparable training
listings actually sold for.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# House specs from the brief's table. Values are best-effort mappings to
# the flag taxonomy in text_features.py -- document the mapping choices
# in Section F, especially flag_ceiling_fitted / flag_no_ceiling which
# are BOTH set to 1 (fitted in bedrooms/lounge, not in kitchen/passage)
# because a binary flag cannot represent partial state cleanly.
BASE_HOUSE = {
    # structured
    "bedrooms": 4,
    "bathrooms": 2,
    "land_area_m2": 300,
    # text flags (Section 3.1 taxonomy)
    "flag_unpainted": 1,          "flag_painted": 0,
    "flag_unpaved": 1,            "flag_paved": 0,
    "flag_unfenced": 1,           "flag_walled": 0,
    "flag_electric_fence": 0,     "flag_security_24hr": 0,
    "flag_no_ceiling": 1,         "flag_ceiling_fitted": 1,
    "flag_kitchen_unfitted": 1,   "flag_kitchen_fitted": 0,
    "flag_screeded_bare_floor": 1,"flag_tiled": 0,
    "flag_borehole": 0,           "flag_water_tank": 0,
    "flag_municipal_only": 1,
    "flag_solar": 0,              "flag_zesa_good": 1,
    "flag_ensuite": 0,            "flag_bics": 0,
    "flag_staff_quarters": 0,
    "flag_construction_stage": 1, "construction_stage_ordinal": 2,
}
HOUSE_A = {**BASE_HOUSE, "ref": "House_A_Madokero",
            "suburb": "Madokero",
            "flag_title_deeds": 1, "flag_cession": 0}
HOUSE_B = {**BASE_HOUSE, "ref": "House_B_Mabvazuva",
            "suburb": "Mabvazuva",
            "flag_title_deeds": 0, "flag_cession": 1}


def align_house_to_model(house: dict, numeric_cols: list[str],
                          categorical_cols: list[str], train_df: pd.DataFrame
                          ) -> tuple[dict, list[str]]:
    """Fill any feature the model uses but the brief didn't specify with
    the training median. Returns (complete_house, list_of_filled_cols)."""
    h = dict(house)
    filled = []
    for col in numeric_cols:
        if col not in h or pd.isna(h[col]):
            med = train_df[col].median()
            h[col] = float(med) if pd.notna(med) else 0.0
            filled.append(col)
    for col in categorical_cols:
        if col not in h or pd.isna(h[col]):
            mode = train_df[col].mode()
            h[col] = str(mode.iloc[0]) if len(mode) else "unknown"
            filled.append(col)
    return h, filled


def predict_price(model, model_name: str, house: dict,
                   feature_cols: list[str]) -> float:
    if model_name == "GLM_baseline":
        glm_cols = ["suburb", "bedrooms", "bathrooms", "land_area_m2"]
        X = pd.DataFrame([{k: house.get(k) for k in glm_cols}])
        log_pred = float(model.predict(X)[0])
        smear = float(np.mean(np.exp(model.resid)))
        return float(np.exp(log_pred) * smear)
    X = pd.DataFrame([house])[feature_cols]
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str)
    return float(np.exp(model.predict(X)[0]))


def extrapolation_diagnostics(house: dict, train_df: pd.DataFrame,
                                numeric_cols: list[str], k: int = 20) -> dict:
    """Three complementary checks:
    (a) kNN distance in standardised numeric feature space;
    (b) count of training rows whose 'salient' flag values match the house
        (unpainted + unfitted kitchen + screeded floor + unfenced), by
        suburb and overall;
    (c) median price of the k nearest training listings -- a LOCAL
        comparator for the model's point estimate.
    """
    # (a) kNN distance
    avail = [c for c in numeric_cols if c in train_df.columns]
    Xtr = train_df[avail].fillna(train_df[avail].median())
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    x_new = np.array([house.get(c, np.nan) for c in avail], dtype=float)
    x_new = np.where(np.isnan(x_new),
                      train_df[avail].median().values, x_new)
    x_new_s = scaler.transform(x_new.reshape(1, -1))
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr_s))).fit(Xtr_s)
    dists, idxs = nn.kneighbors(x_new_s)

    # (c) median price of those neighbours
    neighbour_prices = train_df.iloc[idxs[0]]["price_usd"].dropna()
    local_median = float(neighbour_prices.median()) if len(neighbour_prices) else np.nan

    # (b) prevalence of the "salient" finish pattern
    salient = ["flag_unpainted", "flag_kitchen_unfitted",
                "flag_screeded_bare_floor", "flag_unfenced"]
    salient = [c for c in salient if c in train_df.columns]
    if salient:
        mask = np.ones(len(train_df), dtype=bool)
        for c in salient:
            mask &= (train_df[c].fillna(0).astype(float) == 1)
        n_match = int(mask.sum())
        n_match_suburb = int((mask &
                              (train_df["suburb"] == house["suburb"])).sum())
    else:
        n_match = n_match_suburb = 0

    return {
        "knn_k": k,
        "knn_distance_to_kth":      float(dists[0][-1]),
        "knn_distance_to_1st":      float(dists[0][0]),
        "n_neighbourhood_matching_salient_pattern": n_match,
        "n_in_target_suburb_matching_salient_pattern": n_match_suburb,
        "local_median_price_of_knn_usd": local_median,
        "neighbour_refs": train_df.iloc[idxs[0]]["ref"].tolist()[:5],
    }


def suburb_finish_profile(train_df: pd.DataFrame, suburb: str) -> dict:
    """Distribution of the finish-level flags within a target suburb, so
    you can say quantitatively how atypical the House A/B specification
    is FOR THAT SUBURB -- this is what the brief asks when it warns both
    houses 'fall outside the typical finish level'."""
    sub = train_df[train_df["suburb"] == suburb]
    n = len(sub)
    if n == 0:
        return {"suburb": suburb, "n_in_training": 0}
    out = {"suburb": suburb, "n_in_training": n}
    for c in ["flag_unpainted", "flag_painted", "flag_kitchen_fitted",
              "flag_kitchen_unfitted", "flag_borehole", "flag_walled",
              "flag_tiled"]:
        if c in sub.columns:
            out[f"pct_{c}"] = float(sub[c].fillna(0).astype(float).mean() * 100)
    if "construction_stage_ordinal" in sub.columns:
        out["median_construction_stage"] = float(
            sub["construction_stage_ordinal"].median())
    out["median_price_usd"] = float(sub["price_usd"].median())
    return out


def decompose_gap(model, model_name, house_a, house_b, feature_cols):
    """Two-path decomposition. Non-additive models give different answers
    depending on the order you change the two variables; reporting both
    is the honest way to represent the A-B gap. The difference between
    paths is the suburb x title interaction the model encodes."""
    pred_a = predict_price(model, model_name, house_a, feature_cols)
    pred_b = predict_price(model, model_name, house_b, feature_cols)

    # Path 1: change suburb first (House A -> as if located in Mabvazuva,
    # holding title=deeds), then change title
    a_as_b_suburb = {**house_a, "suburb": house_b["suburb"]}
    pred_a_in_b = predict_price(model, model_name, a_as_b_suburb, feature_cols)
    location_effect_p1 = pred_a - pred_a_in_b
    title_effect_p1    = pred_a_in_b - pred_b

    # Path 2: change title first (House A -> deeds kept, as if cession),
    # then change suburb
    a_with_cession = {**house_a, "flag_title_deeds": 0, "flag_cession": 1}
    pred_a_cession = predict_price(model, model_name, a_with_cession, feature_cols)
    title_effect_p2    = pred_a - pred_a_cession
    location_effect_p2 = pred_a_cession - pred_b

    return {
        "pred_house_A_usd":           pred_a,
        "pred_house_B_usd":           pred_b,
        "total_gap_usd":              pred_a - pred_b,
        "location_effect_path1_usd":  location_effect_p1,
        "title_effect_path1_usd":     title_effect_p1,
        "title_effect_path2_usd":     title_effect_p2,
        "location_effect_path2_usd":  location_effect_p2,
        "interaction_residual_usd":   (location_effect_p1 + title_effect_p1)
                                        - (location_effect_p2 + title_effect_p2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  default="/content/model_ready_features.csv")
    p.add_argument("--model-dir", default="/content/models")
    p.add_argument("--model-name", default="xgboost",
                    help="One of GLM_baseline, random_forest, xgboost, "
                         "elasticnet_interactions.")
    p.add_argument("--outdir", default="/content/case_study")
    p.add_argument("--knn-k", type=int, default=20)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.model_dir)

    df = pd.read_csv(args.features)
    splits = pd.read_csv(mdir / "split_indices.csv")
    df = df.merge(splits, on="ref", how="left")
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    cols = json.load(open(mdir / "feature_cols.json"))
    halfwidths = json.load(open(mdir / "conformal_halfwidths.json"))
    model = joblib.load(mdir / f"{args.model_name}.joblib")
    hw = float(halfwidths[args.model_name])
    feature_cols = cols["numeric"] + cols["categorical"]

    print(f"Model: {args.model_name}")
    print(f"Conformal half-width: USD {hw:,.0f} "
          f"({int((1-cols['alpha'])*100)}% prediction interval)")
    print(f"Training set: n={len(train_df)}")

    # Align both houses to the model's feature list
    house_a, filled_a = align_house_to_model(
        HOUSE_A, cols["numeric"], cols["categorical"], train_df)
    house_b, filled_b = align_house_to_model(
        HOUSE_B, cols["numeric"], cols["categorical"], train_df)
    if filled_a:
        print(f"\nHouse A: filled {len(filled_a)} unspecified feature(s) with "
              f"training median:\n  {filled_a}")
    if filled_b:
        print(f"\nHouse B: filled {len(filled_b)} unspecified feature(s) with "
              f"training median:\n  {filled_b}")

    # ---- point estimates + PIs ----
    pred_a = predict_price(model, args.model_name, house_a, feature_cols)
    pred_b = predict_price(model, args.model_name, house_b, feature_cols)

    print("\n" + "=" * 72)
    print(f"SECTION F VALUATIONS ({args.model_name})")
    print("=" * 72)
    print(f"{'House':35s}  {'Point':>12s}  {'Interval (USD)':>30s}")
    print(f"{'House A (Madokero, deeds)':35s}  {pred_a:>12,.0f}  "
          f"{pred_a-hw:>13,.0f} - {pred_a+hw:<13,.0f}")
    print(f"{'House B (Mabvazuva, cession)':35s}  {pred_b:>12,.0f}  "
          f"{pred_b-hw:>13,.0f} - {pred_b+hw:<13,.0f}")
    print(f"\nLocation + title gap (A - B): USD {pred_a - pred_b:,.0f} "
          f"({(pred_a - pred_b)/pred_b*100:.1f}% of House B)")

    # ---- extrapolation diagnostics ----
    print("\n--- Extrapolation diagnostics ---")
    diag_a = extrapolation_diagnostics(house_a, train_df, cols["numeric"],
                                         k=args.knn_k)
    diag_b = extrapolation_diagnostics(house_b, train_df, cols["numeric"],
                                         k=args.knn_k)
    for tag, d in [("House A", diag_a), ("House B", diag_b)]:
        print(f"\n{tag}:")
        print(f"  kNN distance to nearest training listing: "
              f"{d['knn_distance_to_1st']:.2f}")
        print(f"  kNN distance to {d['knn_k']}-th nearest: "
              f"{d['knn_distance_to_kth']:.2f}")
        print(f"  Training listings matching the 'unfinished' salient pattern "
              f"(unpainted + unfitted kitchen + screeded floor + unfenced): "
              f"{d['n_neighbourhood_matching_salient_pattern']} overall, "
              f"{d['n_in_target_suburb_matching_salient_pattern']} in the "
              f"target suburb")
        print(f"  Median price of the {d['knn_k']} nearest training listings: "
              f"USD {d['local_median_price_of_knn_usd']:,.0f}")
        print(f"  e.g. refs: {d['neighbour_refs']}")

    # ---- suburb finish profile ----
    print("\n--- Finish profile of each target suburb (in training data) ---")
    prof_a = suburb_finish_profile(train_df, "Madokero")
    prof_b = suburb_finish_profile(train_df, "Mabvazuva")
    for tag, pr in [("Madokero", prof_a), ("Mabvazuva", prof_b)]:
        print(f"\n{tag}:  n={pr.get('n_in_training', 0)}")
        if pr.get("n_in_training", 0) > 0:
            for k_ in sorted(pr):
                if k_ not in ("suburb", "n_in_training"):
                    v = pr[k_]
                    if isinstance(v, float):
                        print(f"  {k_:35s} = {v:,.2f}")
                    else:
                        print(f"  {k_:35s} = {v}")

    # ---- two-path gap decomposition ----
    decomp = decompose_gap(model, args.model_name, house_a, house_b, feature_cols)
    print("\n--- Two-path decomposition of the A-B gap ---")
    for k_, v in decomp.items():
        if isinstance(v, float):
            print(f"  {k_:35s} = USD {v:>12,.0f}")
        else:
            print(f"  {k_:35s} = {v}")
    print("""
Interpretation guide:
  - If the two paths give materially different answers, the model encodes
    a suburb x title interaction; report both numbers and the difference.
  - 'interaction_residual_usd' is that difference. If it is small relative
    to the location/title effects, the model is approximately additive and
    a single decomposition is defensible (state which path you'd quote).
  - The SHAP location-share from Section E should be roughly consistent
    with the location/total ratio here; note any tension explicitly.""")

    # ---- summary CSV + plot ----
    summary = pd.DataFrame([
        {"house": "House_A_Madokero", "suburb": "Madokero",
         "title": "deeds", "point_usd": pred_a,
         "pi_lower_usd": pred_a - hw, "pi_upper_usd": pred_a + hw,
         "knn_dist": diag_a["knn_distance_to_kth"],
         "knn_median_usd": diag_a["local_median_price_of_knn_usd"],
         "n_match_salient_in_suburb": diag_a["n_in_target_suburb_matching_salient_pattern"]},
        {"house": "House_B_Mabvazuva", "suburb": "Mabvazuva",
         "title": "cession", "point_usd": pred_b,
         "pi_lower_usd": pred_b - hw, "pi_upper_usd": pred_b + hw,
         "knn_dist": diag_b["knn_distance_to_kth"],
         "knn_median_usd": diag_b["local_median_price_of_knn_usd"],
         "n_match_salient_in_suburb": diag_b["n_in_target_suburb_matching_salient_pattern"]},
    ])
    summary.to_csv(outdir / "case_study_valuations.csv", index=False)

    with open(outdir / "case_study_diagnostics.json", "w") as f:
        json.dump({"model": args.model_name,
                    "halfwidth_usd": hw,
                    "house_A": {**{k: (v if not isinstance(v, np.generic) else v.item())
                                    for k, v in house_a.items()}},
                    "house_B": {**{k: (v if not isinstance(v, np.generic) else v.item())
                                    for k, v in house_b.items()}},
                    "diagnostics_A": diag_a, "diagnostics_B": diag_b,
                    "decomposition": decomp,
                    "suburb_profile_A": prof_a, "suburb_profile_B": prof_b},
                    f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [0, 1]
    ys = [pred_a, pred_b]
    errs = [hw, hw]
    bars = ax.bar(xs, ys, yerr=errs, capsize=12,
                    color=["#3b7dd8", "#d87b3b"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"House A\nMadokero (deeds)\nn={prof_a.get('n_in_training', 0)}",
                          f"House B\nMabvazuva (cession)\nn={prof_b.get('n_in_training', 0)}"])
    for x, y, bar in zip(xs, ys, bars):
        ax.text(x, y, f"  USD {y:,.0f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Modelled market value (USD)")
    ax.set_title(f"Section F valuations with {int((1-cols['alpha'])*100)}% "
                  f"prediction intervals ({args.model_name})")
    fig.tight_layout()
    fig.savefig(outdir / "case_study_valuations.png", dpi=150)
    plt.close(fig)

    print(f"\nAll Section F outputs written to {outdir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
text_features.py  --  Section 3.1 (text feature engineering)
Reads /content/cleaned_listings.csv, writes /content/text_features.csv
plus three side-car CSVs used for the Section 3.1 write-up.
"""
from __future__ import annotations
import argparse, re, json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

VOCABULARY = {
    "flag_unpainted": ("paint", [
        r"\bunpainted\b", r"\bbare\s+(plaster|cement|brick)\b",
        r"\broughcast\b", r"\bwall[- ]plate\b",
        r"finish\s+to\s+your\s+own\s+taste",
    ], None),
    "flag_painted": ("paint", [
        r"\bpainted\b(?!.{0,15}unpainted)", r"\bfreshly\s+painted\b",
        r"\bnewly\s+painted\b",
    ], None),
    "flag_paved": ("paving", [r"\bpaved\b", r"\bpaving\b", r"\btarred\s+driveway\b"], None),
    "flag_unpaved": ("paving", [r"\bunpaved\b", r"\bbare\s+earth\b", r"\bdirt\s+driveway\b"], None),
    "flag_walled": ("security", [r"\bwalled\b", r"\bdurawall\b", r"\bdurawalled\b",
                                   r"\bwalled\s+and\s+gated\b"], None),
    "flag_unfenced": ("security", [r"\bunfenced\b", r"\bnot\s+yet\s+walled\b",
                                     r"\bno\s+(wall|durawall|fence)\b"], None),
    "flag_electric_fence": ("security", [r"\belectric\s+fence\b"], None),
    "flag_security_24hr": ("security", [r"24\s?hr\s+security", r"\bgated\s+community\b"], None),
    "flag_ceiling_fitted": ("ceiling", [r"\bceiling(?:s)?\s+fitted\b", r"\bceiled\b"], None),
    "flag_no_ceiling": ("ceiling", [r"\bno\s+ceiling\b", r"\btrusses\s+exposed\b",
                                      r"roof\s+underside\s+exposed"], None),
    "flag_kitchen_fitted": ("kitchen", [r"\bfitted\s+kitchen\b", r"\bgourmet\s+kitchen\b",
                                          r"\bmodern\s+kitchen\b"], None),
    "flag_kitchen_unfitted": ("kitchen", [
        r"\bnot\s+fitted\b.{0,20}kitchen|\bkitchen\b.{0,20}not\s+fitted",
        r"\bstub[- ]outs?\s+only\b", r"\bplumbing\s+and\s+electrical\s+stub"], None),
    "flag_tiled": ("flooring", [r"\btiled\b", r"\bfully\s+tiled\b"], None),
    "flag_screeded_bare_floor": ("flooring", [r"\bscreeded\b", r"\bbare\s+cement\s+floor",
                                                r"\bcement\s+floors?\b(?!.{0,10}tiled)"], None),
    "flag_borehole": ("water", [r"\bborehole\b"], None),
    "flag_water_tank": ("water", [r"\bwater\s+tank\b",
                                    r"\d\s?000\s?l(?:tr|itre)?\s+tank"], None),
    "flag_municipal_only": ("water", [r"\bmunicipal\s+(?:water|mains)\s+only\b",
                                        r"\bcouncil\s+water\s+only\b"], None),
    "flag_solar": ("power", [r"\bsolar\b", r"\bsolar[- ]powered\b",
                               r"\d(?:\.\d)?\s?kva\s+solar"], None),
    "flag_zesa_good": ("power", [r"\bgood\s+zesa\b", r"\bzesa[- ]connected\b",
                                   r"\bprepaid\s+meter\b"], None),
    "flag_ensuite": ("layout", [r"\ben[- ]?suite\b", r"\bmain\s+en\s?suite\b",
                                  r"\bmes\b"], None),
    "flag_bics": ("layout", [r"\bbics?\b", r"built[- ]?in\s+cupboards?"], None),
    "flag_staff_quarters": ("layout", [r"\bstaff\s+quarters\b", r"\bcottage\b"], None),
    "flag_construction_stage": ("construction_stage", [
        r"\bshell\b", r"\bincomplete\b", r"\bwall[- ]plate\s+level\b",
        r"\bstructurally\s+complete\b", r"\broofed\b(?!.{0,20}complete)",
        r"\bmove[- ]in\s+ready\b", r"\bfully\s+finished\b", r"\bturn[- ]?key\b",
    ], "construction_stage_ordinal"),
    "flag_title_deeds": ("title", [r"\btitle\s+deeds?\b"], None),
    "flag_cession": ("title", [r"\bcession\b", r"\bdeed\s+of\s+cession\b",
                                 r"\bconsent\s+to\s+sell\b"], None),
}

CONSTRUCTION_STAGE_PATTERNS = [
    (3, [r"\bmove[- ]in\s+ready\b", r"\bfully\s+finished\b", r"\bturn[- ]?key\b",
         r"\bimmaculate\b"]),
    (2, [r"\bstructurally\s+complete\b", r"\bfinish\s+to\s+your\s+own\s+taste\b",
         r"\broofed,?\s+windows?\s+and\s+doors?\s+fitted\b"]),
    (1, [r"\bshell\b", r"\bincomplete\b", r"\bwall[- ]plate\s+level\b"]),
]


def apply_vocabulary(df, text_col="description"):
    text = df[text_col].fillna("").str.lower()
    flag_cols = []
    for flag_name, (category, patterns, ordinal_key) in VOCABULARY.items():
        if ordinal_key:
            continue
        combined = re.compile("|".join(patterns))
        df[flag_name] = text.apply(lambda t: bool(combined.search(t)))
        flag_cols.append(flag_name)
    def stage_score(t):
        for score, patterns in CONSTRUCTION_STAGE_PATTERNS:
            if any(re.search(p, t) for p in patterns):
                return score
        return 0
    df["construction_stage_ordinal"] = text.apply(stage_score)
    flag_cols.append("construction_stage_ordinal")
    return df, flag_cols


def sample_verbatim_matches(df, text_col="description", n_per_flag=2):
    text = df[text_col].fillna("")
    rows = []
    for flag_name, (category, patterns, _) in VOCABULARY.items():
        if flag_name not in df.columns:
            continue
        combined = re.compile("|".join(patterns), re.IGNORECASE)
        matched = df[df[flag_name] == True] if df[flag_name].dtype == bool else pd.DataFrame()
        count = 0
        for _, row in matched.iterrows():
            m = combined.search(str(row[text_col]))
            if m:
                start = max(0, m.start() - 30); end = min(len(row[text_col]), m.end() + 30)
                snippet = row[text_col][start:end].strip()
                rows.append({"flag": flag_name, "category": category,
                              "ref": row.get("ref", ""),
                              "verbatim_snippet": f"...{snippet}..."})
                count += 1
            if count >= n_per_flag:
                break
    return pd.DataFrame(rows)


def suggest_vocabulary_additions(df, text_col="description", top_n=40):
    text = df[text_col].fillna("").str.lower()
    already_covered = re.compile("|".join(
        p for _, patterns, _ in VOCABULARY.values() for p in patterns))
    vec = TfidfVectorizer(ngram_range=(2, 3), stop_words="english",
                           min_df=3, max_features=2000)
    try:
        vec.fit(text)
    except ValueError:
        return pd.DataFrame(columns=["phrase", "doc_frequency"])
    doc_counts = Counter()
    for doc in text:
        for term in set(vec.build_analyzer()(doc)):
            doc_counts[term] += 1
    candidates = [(t, f) for t, f in doc_counts.items() if not already_covered.search(t)]
    candidates.sort(key=lambda x: -x[1])
    return pd.DataFrame(candidates[:top_n], columns=["phrase", "doc_frequency"])


def tfidf_features(df, text_col="description", k_best=50, target_col="price_usd"):
    text = df[text_col].fillna("")
    y = np.log(df[target_col].clip(lower=1))
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                           min_df=5, max_df=0.6, max_features=1000)
    X = vec.fit_transform(text)
    k = min(k_best, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X.toarray(), y)
    selected_terms = np.array(vec.get_feature_names_out())[selector.get_support()]
    return X_selected, list(selected_terms)


def compare_marginal_contribution(df, flag_cols, tfidf_X,
                                   target_col="price_usd", n_folds=5):
    y = np.log(df[target_col].clip(lower=1)).values
    flags_X = df[flag_cols].astype(float).values
    def cv_r2(X):
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        return scores.mean(), scores.std()
    return {
        "rule_flags_only":   cv_r2(flags_X),
        "tfidf_only":        cv_r2(tfidf_X),
        "flags_plus_tfidf":  cv_r2(np.hstack([flags_X, tfidf_X])),
    }


def suburb_language_consistency(df, flag_cols, suburb_col="suburb", min_listings=10):
    counts = df.groupby(suburb_col).size()
    valid_suburbs = counts[counts >= min_listings].index
    subset = df[df[suburb_col].isin(valid_suburbs)]
    rows = []
    for flag in flag_cols:
        if df[flag].dtype == bool:
            by_suburb = subset.groupby(suburb_col)[flag].mean() * 100
        else:
            by_suburb = subset.groupby(suburb_col)[flag].mean()
        rows.append({
            "flag": flag,
            "min_suburb_rate": by_suburb.min(),
            "max_suburb_rate": by_suburb.max(),
            "spread": by_suburb.max() - by_suburb.min(),
            "suburb_with_min": by_suburb.idxmin(),
            "suburb_with_max": by_suburb.idxmax(),
        })
    return pd.DataFrame(rows).sort_values("spread", ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path, default=Path("/content/cleaned_listings.csv"))
    p.add_argument("--output", type=Path, default=Path("/content/text_features.csv"))
    p.add_argument("--outdir", type=Path, default=Path("/content"))
    args = p.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} cleaned listings")

    df, flag_cols = apply_vocabulary(df)
    print(f"Applied {len(flag_cols)} vocabulary flags")

    verbatim = sample_verbatim_matches(df)
    verbatim.to_csv(args.outdir / "vocabulary_verbatim_evidence.csv", index=False)
    print(f"Wrote {len(verbatim)} verbatim snippets to "
          f"{args.outdir / 'vocabulary_verbatim_evidence.csv'}")

    suggestions = suggest_vocabulary_additions(df)
    suggestions.to_csv(args.outdir / "vocabulary_suggestions.csv", index=False)
    print(f"Wrote {len(suggestions)} candidate phrases to "
          f"{args.outdir / 'vocabulary_suggestions.csv'}")

    if "price_usd" in df.columns and df["price_usd"].notna().sum() > 20:
        tfidf_X, terms = tfidf_features(df)
        print(f"TF-IDF selected {len(terms)} terms: {terms[:15]}...")
        results = compare_marginal_contribution(df, flag_cols, tfidf_X)
        print("\nMarginal contribution (5-fold CV R^2 on log price):")
        for name, (m, s) in results.items():
            print(f"  {name:20s}: R^2 = {m:.3f} (+/- {s:.3f})")

    consistency = suburb_language_consistency(df, flag_cols)
    consistency.to_csv(args.outdir / "suburb_language_consistency.csv", index=False)
    print(f"\nWrote {args.outdir / 'suburb_language_consistency.csv'}")

    df.to_csv(args.output, index=False)
    print(f"\nWrote full text-feature table ({len(df.columns)} columns) to {args.output}")


if __name__ == "__main__":
    main()

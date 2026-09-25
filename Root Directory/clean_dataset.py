#!/usr/bin/env python3
"""
clean_dataset.py
=================
Section 2.4 (Assignment 1): cleans /content/Raw data.csv into a
model-ready dataset, with every decision logged rather than applied
silently -- the brief explicitly penalises "silently drop every row with
a missing field", so this script writes a decisions log alongside the
cleaned CSV, and separates "removed" from "flagged" (flagged rows are
kept, with a boolean column, so you can inspect and decide in your memo).

WHY A SEPARATE STEP FROM THE SCRAPER
--------------------------------------
Cleaning decisions (which outliers to cut, how to impute, which FX rate
to use) are exactly the kind of thing a marker wants to see you reason
about explicitly, in your own words, in the data-collection-and-ethics
memo. Keeping this as its own script/notebook cell (rather than baking
it into the scraper) makes each decision inspectable and re-runnable
independently -- e.g. if you decide midway through your memo that a
different FX date is more defensible, you re-run this one script, not
the whole multi-hour scrape.

USAGE (Colab)
-------------
    %%writefile clean_dataset.py
    # (this whole file)

    !python clean_dataset.py \
        --fx-rate 26.8 --fx-date 2026-09-20 \
        --fx-source "RBZ official interbank mid-rate, https://www.rbz.co.zw"

Or, if you prefer to be explicit about paths:
    !python clean_dataset.py \
        --input "/content/Raw data.csv" \
        --output "/content/cleaned_listings.csv" \
        --fx-rate 26.8 --fx-date 2026-09-20 \
        --fx-source "RBZ official interbank mid-rate, https://www.rbz.co.zw"

You MUST supply --fx-rate/--fx-date/--fx-source yourself, looked up on
(or as close as possible to) the day you actually scraped -- I'm not
hardcoding a number here because a stale or unsourced FX figure is
worse than the script refusing to run. Check https://www.rbz.co.zw for
the official interbank rate on your scrape date; note the parallel
("street") rate is a different, contested number -- state in your memo
which one you used and why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------

REQUIRED_COLS = ["ref", "url", "price_value", "currency", "suburb",
                  "bedrooms", "bathrooms", "land_area_m2", "description"]

# Your CSV's column names -> the internal names used everywhere below.
# Keeping this mapping in one place means if the scraper's output schema
# changes (e.g. a column renamed), you only edit it here.
CSV_RENAME_MAP = {
    "listing_id":        "ref",
    "listing_url":       "url",
    "advertised_price":  "price_value",
    "address_locality":  "suburb",
    "stand_size":        "land_area_m2",
    "floor_area":        "floor_area_m2",
    "agency":            "agency_name",
}

ACRE_TO_M2 = 4046.8564224


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=CSV_RENAME_MAP)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"WARNING: expected columns missing from raw data: {missing_cols}",
              file=sys.stderr)

    # Coerce numeric columns -- pandas reads blank cells as NaN already,
    # but if the scraper ever emits "$14,000" or "n/a" we want that to
    # become NaN rather than raise on arithmetic downstream.
    for col in ["price_value", "bedrooms", "bathrooms",
                 "land_area_m2", "floor_area_m2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalise_units(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Convert any acre-denominated areas to m² so every area feature in
    the dataset shares one unit. Your CSV has stand_size_unit in {m²,
    Acres, blank}, and one Philadelphia listing whose stand is quoted as
    5.4 acres -- without this step that row would look like a 5.4 m²
    stand and distort every size-based summary."""
    for value_col, unit_col in [("land_area_m2", "stand_size_unit"),
                                 ("floor_area_m2", "floor_area_unit")]:
        if value_col not in df.columns or unit_col not in df.columns:
            continue
        is_acres = (df[unit_col].astype(str).str.lower()
                     .str.contains("acre", na=False))
        n = int(is_acres.sum())
        if n:
            df.loc[is_acres, value_col] = (
                df.loc[is_acres, value_col] * ACRE_TO_M2)
            log.append(f"{value_col}: converted {n} value(s) from acres "
                        f"to m² (1 acre = {ACRE_TO_M2:.2f} m²), because a "
                        f"5.4-acre stand stored as '5.4 m²' would be a "
                        f"~40,000x understatement and skew every area "
                        f"statistic.")
    return df


# --------------------------------------------------------------------------
# 1. De-duplication
# --------------------------------------------------------------------------

def dedupe(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["ref"], keep="last")
    log.append(f"Exact ref duplicates removed: {before - len(df)} "
                f"(kept the LAST-scraped copy, on the assumption a later "
                f"scrape reflects a more current price/description)")

    # Re-listed properties: same agency + suburb + bedrooms + bathrooms +
    # land area + a price within 2% of each other are almost certainly the
    # same physical property re-advertised under a new ref. We do NOT
    # auto-drop these silently -- we flag them and keep the most recently
    # updated one, because re-listing timing is itself informative (e.g.
    # a property re-listed after a price cut).
    dupe_key_cols = ["agency_name", "suburb", "bedrooms",
                     "bathrooms", "land_area_m2"]
    available_key_cols = [c for c in dupe_key_cols if c in df.columns]
    df["likely_relisted_duplicate"] = False
    if available_key_cols and "price_value" in df.columns:
        df["_dupe_key"] = (df[available_key_cols].fillna("")
                            .astype(str).agg("|".join, axis=1))
        flagged_relist_refs: set = set()
        for _, group in df.groupby("_dupe_key"):
            if len(group) < 2:
                continue
            prices = group["price_value"].dropna()
            if len(prices) < 2:
                continue
            if prices.max() > 0 and (prices.max() - prices.min()) / prices.max() < 0.02:
                flagged_relist_refs.update(group["ref"].tolist())
        df["likely_relisted_duplicate"] = df["ref"].isin(flagged_relist_refs)
        log.append(f"Listings flagged as likely re-listed duplicates "
                    f"(same agency/suburb/beds/baths/land area, price "
                    f"within 2%): {len(flagged_relist_refs)}. NOT "
                    f"auto-removed -- kept with a flag column; decide "
                    f"in your memo whether to drop, and justify it "
                    f"with 2-3 example pairs.")
        df = df.drop(columns=["_dupe_key"])
    return df


# --------------------------------------------------------------------------
# 2. Currency standardisation
# --------------------------------------------------------------------------

def standardise_currency(df: pd.DataFrame, fx_rate: float, fx_date: str,
                          fx_source: str, log: list[str]) -> pd.DataFrame:
    df["price_usd"] = np.nan
    df["fx_rate_applied"] = np.nan
    df["fx_assumption"] = ""

    cur = df["currency"].astype(str).str.upper().str.strip()
    is_zig = cur.isin(["ZIG", "Z$", "ZWG"])
    is_usd = cur.isin(["USD", "US$", "$"])
    is_unknown = ~(is_zig | is_usd)

    df.loc[is_zig, "price_usd"] = df.loc[is_zig, "price_value"] / fx_rate
    df.loc[is_zig, "fx_rate_applied"] = fx_rate
    df.loc[is_zig, "fx_assumption"] = f"ZiG->USD @ {fx_rate} ({fx_source}, {fx_date})"

    df.loc[is_usd, "price_usd"] = df.loc[is_usd, "price_value"]
    df.loc[is_usd, "fx_assumption"] = "already USD"

    # property.co.zw's default page rendering is USD-denominated unless a
    # ZiG toggle is explicitly reflected in the scraped text; a listing
    # with no currency symbol captured is assumed USD, but this is an
    # ASSUMPTION, not a fact -- flag it so you can spot check a sample by
    # hand and say so explicitly in your memo rather than asserting it.
    df.loc[is_unknown, "price_usd"] = df.loc[is_unknown, "price_value"]
    df.loc[is_unknown, "fx_assumption"] = ("no currency marker scraped -- "
                                            "ASSUMED USD, unverified")

    log.append(f"Currency standardisation: {is_zig.sum()} listings "
                f"converted ZiG->USD at {fx_rate} (source: {fx_source}, "
                f"{fx_date}); {is_usd.sum()} already USD; {is_unknown.sum()} "
                f"had no currency marker and were ASSUMED USD (spot-check "
                f"these by hand before trusting them -- see 'fx_assumption' "
                f"column).")
    return df


# --------------------------------------------------------------------------
# 3. Missing values
# --------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    """Documented, field-by-field strategy. The brief penalises blanket
    row-dropping, so: only drop rows missing something we cannot proceed
    without (price, suburb); for everything else, impute or flag."""

    before = len(df)
    df = df.dropna(subset=["price_usd", "suburb"])
    dropped_critical = before - len(df)
    log.append(f"Dropped {dropped_critical} rows missing price or suburb "
                f"(both are required for every downstream section -- "
                f"EDA, GLM, ML, the Section F case study -- so a row "
                f"without them cannot be used, not even with imputation).")

    # bedrooms/bathrooms: impute with suburb median (more defensible than
    # global median, since a 1-bed in Borrowdale and a 1-bed in Budiriro
    # are different populations), flag imputed rows rather than hiding it.
    for col in ["bedrooms", "bathrooms"]:
        if col not in df.columns:
            continue
        flag_col = f"{col}_imputed"
        df[flag_col] = df[col].isna()
        n_missing = int(df[col].isna().sum())
        df[col] = df.groupby("suburb")[col].transform(
            lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())  # global fallback for
        # suburbs with zero non-missing values in this column
        log.append(f"{col}: {n_missing} missing values imputed with the "
                    f"suburb median (global median as fallback). Flagged "
                    f"in '{flag_col}' -- consider excluding these from any "
                    f"figure/statistic where imputation would visibly "
                    f"distort the picture (e.g. don't imputed-pad a small "
                    f"suburb sample in the Section C summary table).")

    # land_area_m2 / floor_area_m2: often genuinely absent for townhouses/
    # flats where land area isn't meaningful, and missingness itself may
    # correlate with property type -- do NOT impute a size; instead keep
    # NaN and add a boolean "has_land_area" feature, since "missing" here
    # is informative, not just noise.
    for col in ["land_area_m2", "floor_area_m2"]:
        if col not in df.columns:
            continue
        df[f"has_{col}"] = df[col].notna()
        n_missing = int(df[col].isna().sum())
        log.append(f"{col}: {n_missing} missing values left as NaN (NOT "
                    f"imputed) -- added 'has_{col}' flag instead, since "
                    f"missingness here is plausibly informative (e.g. "
                    f"flats/townhouses rarely quote a stand size) rather "
                    f"than a random gap. Your GLM/ML models should treat "
                    f"this explicitly (e.g. a missing-indicator + median "
                    f"imputation *inside* the model pipeline, not before "
                    f"it, so the train/test split in Section 5.3 stays "
                    f"clean).")

    # description: an empty description is a real absence of text signal,
    # not something to impute -- leave blank, Section 3 text features will
    # naturally score it as "no flags detected" for every rule-based flag.
    if "description" in df.columns:
        n_missing = int(df["description"].isna().sum()
                         + (df["description"] == "").sum())
        df["description"] = df["description"].fillna("")
        log.append(f"description: {n_missing} listings had no scraped "
                    f"description text -- left as empty string ('no signal' "
                    f"is itself the correct value here, not something to "
                    f"impute). These rows will get all-zero text flags in "
                    f"Section 3.1; don't let that silently look like "
                    f"'confirmed absence of every condition issue'.")

    return df


# --------------------------------------------------------------------------
# 4. Outlier / data-entry-error detection (flag, don't silently drop)
# --------------------------------------------------------------------------

def flag_outliers(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    # Work on a reset index so positional flags line up cleanly with the
    # list-of-lists we accumulate reasons into.
    df = df.reset_index(drop=True)
    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def add_flag(mask, reason: str) -> None:
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=df.index)
        mask = mask.fillna(False).astype(bool)
        for i in df.index[mask]:
            reasons[i].append(reason)

    # Domain rules -- values that are almost certainly data-entry errors,
    # independent of the statistical distribution.
    add_flag(df["price_usd"] < 500,
             "price_usd<500 (likely missing digit(s) or deposit/rental "
             "figure mis-scraped as sale price)")

    add_flag(df["bedrooms"] > 15,
             "bedrooms>15 (implausible for a single residential listing; "
             "check for a mis-scraped lodge/guesthouse)")

    if "land_area_m2" in df.columns:
        add_flag(df["land_area_m2"] < 50,
                 "land_area_m2<50 (implausibly small stand for a house)")

    # Statistical outliers on log-price, by suburb (a $50k gap means
    # different things in Budiriro vs Borrowdale) -- IQR rule, flagged not
    # dropped, since a genuine luxury/distressed sale is a real data point.
    log_price = np.log(df["price_usd"].clip(lower=1))

    def iqr_flags(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=s.index)
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (s < lo) | (s > hi)

    stat_outlier = log_price.groupby(df["suburb"]).transform(iqr_flags)
    add_flag(stat_outlier,
             "statistical price outlier within its suburb (log-price, "
             "1.5xIQR rule) -- inspect manually, do not assume error")

    df["outlier_flag"] = ["; ".join(r) for r in reasons]

    n_flagged = int((df["outlier_flag"] != "").sum())
    log.append(f"{n_flagged} rows flagged by outlier rules (see "
                f"'outlier_flag' column for reasons). NONE were "
                f"automatically dropped -- the assignment wants documented, "
                f"justified treatment, so open these rows, decide case by "
                f"case (genuine luxury sale vs data-entry error), and "
                f"record your reasoning + final count removed in your memo.")
    return df


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                         default=Path("/content/Raw data.csv"),
                         help="Path to the scraped CSV "
                              "(default: /content/Raw data.csv)")
    parser.add_argument("--output", type=Path,
                         default=Path("/content/cleaned_listings.csv"),
                         help="Where to write the cleaned CSV "
                              "(default: /content/cleaned_listings.csv)")
    parser.add_argument("--fx-rate", type=float, required=True,
                         help="ZiG per 1 USD, looked up by you for your scrape date")
    parser.add_argument("--fx-date", type=str, required=True,
                         help="Date the FX rate applies to, e.g. 2026-09-20")
    parser.add_argument("--fx-source", type=str, required=True,
                         help="Where you got the rate, e.g. "
                              "'RBZ official interbank, rbz.co.zw'")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found -- did you upload "
              f"'Raw data.csv' to /content/?", file=sys.stderr)
        sys.exit(1)

    log: list[str] = []
    df = load_csv(args.input)
    log.append(f"Loaded {len(df)} raw scraped listings from {args.input}")

    df = normalise_units(df, log)
    df = dedupe(df, log)
    df = standardise_currency(df, args.fx_rate, args.fx_date,
                               args.fx_source, log)
    df = handle_missing(df, log)
    df = flag_outliers(df, log)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    log_path = args.output.with_suffix(".cleaning_log.txt")
    with log_path.open("w", encoding="utf-8") as f:
        f.write("Cleaning decisions log -- Section 2.4\n")
        f.write("=" * 50 + "\n\n")
        for i, entry in enumerate(log, 1):
            f.write(f"{i}. {entry}\n\n")
        f.write(f"\nFinal cleaned dataset: {len(df)} rows -> {args.output}\n")

    print(f"\nWrote {len(df)} cleaned rows to {args.output}")
    print(f"Wrote decisions log to {log_path}")
    print("\nOpen the log and turn it into prose for your Section 2.4 "
          "write-up -- it's written to be paraphrased, not pasted verbatim.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
merge_features.py
==================
Section 3 final deliverable: joins the cleaned listing table, the text
flags, and the (optional) image features on 'ref', writing the merged,
model-ready feature table that eda_diagnostics.py, models.py and the
Section E/F scripts all read from.

Note that text_features.py writes out the FULL cleaned dataframe PLUS the
flag columns -- it does not strip the original columns. So the merge here
is a left-join of image features onto text_features, not a join of three
disparate tables. The script still tolerates the case where you ran
clean_dataset.py but not text_features.py (falls back to cleaned_listings
as the base) or where you couldn't run CLIP (image_features.csv missing
-- image columns are simply absent, and has_image_features is False).

USAGE
-----
    python merge_features.py \
        --cleaned /content/cleaned_listings.csv \
        --text    /content/text_features.csv \
        --image   /content/image_features.csv \
        --output  /content/model_ready_features.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned", default="/content/cleaned_listings.csv")
    p.add_argument("--text",    default="/content/text_features.csv")
    p.add_argument("--image",   default="/content/image_features.csv")
    p.add_argument("--output",  default="/content/model_ready_features.csv")
    p.add_argument("--outdir",  default="/content")
    args = p.parse_args()

    cleaned_path, text_path, image_path = (Path(args.cleaned),
                                            Path(args.text),
                                            Path(args.image))

    # ---- base = text_features if present, else cleaned_listings ----
    if text_path.exists():
        base = pd.read_csv(text_path)
        print(f"Loaded base from {text_path}:  {base.shape}")
    elif cleaned_path.exists():
        base = pd.read_csv(cleaned_path)
        print(f"text_features.csv not found -- using cleaned_listings as base: "
              f"{base.shape}. Flags will be missing; run text_features.py first "
              f"if you want them.")
    else:
        raise SystemExit(f"Neither {text_path} nor {cleaned_path} exists. "
                          f"Run clean_dataset.py and text_features.py first.")

    if "ref" not in base.columns:
        raise SystemExit("base table has no 'ref' column -- cannot merge on ref.")

    # ---- left-join image features if they exist ----
    if image_path.exists():
        img = pd.read_csv(image_path)
        print(f"Loaded image features from {image_path}:  {img.shape}")

        if img["ref"].duplicated().any():
            n_dupes = int(img["ref"].duplicated().sum())
            print(f"WARNING: {n_dupes} duplicate refs in image_features -- "
                  f"keeping the first of each. This shouldn't happen; "
                  f"if it does, check whether image_features.py was run twice "
                  f"with different --sample-size settings and concatenated.")
            img = img.drop_duplicates(subset="ref", keep="first")

        before = len(base)
        base = base.merge(img, on="ref", how="left")
        assert len(base) == before, (
            f"row count changed during join ({before} -> {len(base)}); "
            f"image_features has duplicate refs even after de-dupe.")

        base["has_image_features"] = base["n_images_scored"].notna()
        n_img = int(base["has_image_features"].sum())
        print(f"After left-join: {len(base)} rows, "
              f"{n_img} with image features ({n_img / len(base) * 100:.1f}%)")
    else:
        print(f"image_features.csv not found -- proceeding WITHOUT image "
              f"features. Image columns will be absent; has_image_features = "
              f"False for all rows. Section 3.2 and Section E's text-vs-image "
              f"agreement check both need this file; run image_features.py "
              f"(with a GPU in Colab) before those sections.")
        base["has_image_features"] = False

    # ---- quick integrity checks ----
    print(f"\nFinal shape: {base.shape}")
    print(f"Columns: {len(base.columns)}")
    print(f"  - structured: {sum(1 for c in base.columns if c in ['ref','price_usd','suburb','bedrooms','bathrooms','land_area_m2','floor_area_m2','construction_stage_ordinal'])}")
    print(f"  - text flags: {sum(1 for c in base.columns if c.startswith('flag_'))}")
    print(f"  - image feats: {sum(1 for c in base.columns if c.startswith('img_'))}")

    # ---- write output + per-column summary for the report's appendix ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")

    summary = pd.DataFrame({
        "column":   base.columns,
        "dtype":    base.dtypes.astype(str).values,
        "non_null": base.notna().sum().values,
        "null_pct": (base.isna().mean() * 100).round(2).values,
    })
    summary_path = Path(args.outdir) / "model_ready_column_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} (paste into your report as the "
          f"'merged feature table' evidence)")


if __name__ == "__main__":
    main()

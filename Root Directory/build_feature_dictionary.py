#!/usr/bin/env python3
"""Section 3 deliverable: feature dictionary, generated from the same
VOCABULARY / CONDITION_DIMENSIONS dicts that drive the feature code."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from text_features import VOCABULARY, CONSTRUCTION_STAGE_PATTERNS
from image_features import CONDITION_DIMENSIONS


def build() -> pd.DataFrame:
    rows = []
    for flag_name, (category, patterns, ordinal_key) in VOCABULARY.items():
        if ordinal_key:
            continue
        rows.append({
            "feature_name": flag_name,
            "source": "text",
            "category": category,
            "definition": f"1 if the description matches any of: {', '.join(patterns)}; else 0",
            "construction_method": "Regex match (case-insensitive) against the scraped "
                                    "free-text description; see VOCABULARY in text_features.py",
            "type": "binary",
        })
    rows.append({
        "feature_name": "construction_stage_ordinal",
        "source": "text",
        "category": "construction_stage",
        "definition": "0 = no construction-stage language detected; 1 = shell/incomplete/"
                       "wall-plate level; 2 = structurally complete but unfinished; "
                       "3 = move-in ready/fully finished/turn-key",
        "construction_method": "Highest-scoring pattern match, see "
                                "CONSTRUCTION_STAGE_PATTERNS in text_features.py",
        "type": "ordinal (0-3)",
    })
    rows.append({
        "feature_name": "tfidf_selected_terms (variable set, k of them)",
        "source": "text",
        "category": "nlp_general",
        "definition": "TF-IDF weight (1-2 grams, min_df=5, max_df=0.6) for the k terms "
                       "most associated with log(price) via univariate f_regression",
        "construction_method": "sklearn TfidfVectorizer -> SelectKBest(f_regression); "
                                "see tfidf_features() in text_features.py",
        "type": "continuous (TF-IDF weight)",
    })
    for dim_name, (pos, neg) in CONDITION_DIMENSIONS.items():
        for agg in ("mean", "max"):
            rows.append({
                "feature_name": f"{dim_name}_{agg}",
                "source": "image",
                "category": dim_name.replace("img_", ""),
                "definition": f"{agg.capitalize()} (over the listing's scored photos) "
                               f"CLIP softmax probability of '{pos}' vs '{neg}'",
                "construction_method": "Zero-shot CLIP (openai/clip-vit-base-patch32): "
                                        "cosine similarity between image embeddings "
                                        "and paired prompt text embeddings, softmax over "
                                        "the positive/negative pair",
                "type": "continuous (probability, 0-1)",
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                    default=Path("/content/feature_dictionary.csv"))
    args = p.parse_args()
    df = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} feature definitions to {args.output}")
    print(f"  text features:  {(df['source'] == 'text').sum()}")
    print(f"  image features: {(df['source'] == 'image').sum()}")


if __name__ == "__main__":
    main()

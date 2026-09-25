#!/usr/bin/env python3
"""
image_features.py  --  Section 3.2 (visual feature engineering)
Zero-shot CLIP scoring of listing photos. Needs a GPU: in Colab,
Runtime > Change runtime type > Hardware accelerator > T4 GPU.
"""
from __future__ import annotations
import argparse, io, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests

CONDITION_DIMENSIONS = {
    "img_painted":          ("a photo of a freshly painted interior wall",
                             "a photo of a bare, unpainted plaster or cement wall"),
    "img_paved":            ("a photo of a paved driveway or courtyard",
                             "a photo of a bare earth, unpaved driveway"),
    "img_walled_secure":    ("a photo of a walled or durawalled property boundary",
                             "a photo of an open, unfenced yard with no boundary wall"),
    "img_ceiling_fitted":   ("a photo of a room with a finished ceiling",
                             "a photo of a room with exposed roof trusses and no ceiling"),
    "img_kitchen_fitted":   ("a photo of a fitted kitchen with cabinets and countertops",
                             "a photo of a bare, unfinished kitchen room with no cabinets"),
    "img_tiled_floor":      ("a photo of a tiled floor",
                             "a photo of a bare, unfinished cement or screeded floor"),
    "img_complete_finish":  ("a photo of a fully finished, move-in-ready house exterior",
                             "a photo of an incomplete or under-construction house exterior"),
    "img_garden_landscaped":("a photo of a landscaped, maintained garden",
                             "a photo of an overgrown or bare, unlandscaped yard"),
}


def load_image_bytes(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout,
                          headers={"User-Agent": "MSAC512-CourseworkBot/1.0"})
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        pass
    return None


def score_images_with_clip(image_urls_by_ref, max_images_per_listing=6,
                            model_name="openai/clip-vit-base-patch32", device=None):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device} ...")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    dims = list(CONDITION_DIMENSIONS.keys())
    prompts = []
    for pos, neg in CONDITION_DIMENSIONS.values():
        prompts.extend([pos, neg])

    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        tf = model.get_text_features(**text_inputs)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    records = []
    n = len(image_urls_by_ref)
    for i, (ref, urls) in enumerate(image_urls_by_ref.items(), start=1):
        urls = urls[:max_images_per_listing]
        per_image_scores = []
        for url in urls:
            b = load_image_bytes(url)
            if b is None:
                continue
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                continue
            image_inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                img_feat = model.get_image_features(**image_inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * img_feat @ tf.T).squeeze(0)
            dim_scores = []
            for d in range(len(dims)):
                pair = logits[2*d : 2*d + 2]
                dim_scores.append(torch.softmax(pair, dim=0)[0].item())
            per_image_scores.append(dim_scores)
        if not per_image_scores:
            continue
        arr = np.array(per_image_scores)
        row = {"ref": ref, "n_images_scored": len(per_image_scores)}
        for d_idx, dim_name in enumerate(dims):
            row[f"{dim_name}_mean"] = arr[:, d_idx].mean()
            row[f"{dim_name}_max"]  = arr[:, d_idx].max()
        records.append(row)
        if i % 10 == 0 or i == n:
            print(f"  scored {i}/{n} listings")
    return pd.DataFrame(records)


def generate_manual_audit_template(df, ref_col, image_col, sample_size, out_path,
                                    double_code_n=20, seed=42):
    rng = np.random.default_rng(seed)
    has_images = df[image_col].fillna("").astype(str).str.len() > 0
    pool = df[has_images]
    sample = pool.sample(n=min(sample_size, len(pool)), random_state=seed)
    double_coded_refs = set(sample.sample(n=min(double_code_n, len(sample)),
                                            random_state=seed)[ref_col])
    rows = []
    dims = [d.replace("img_", "") for d in CONDITION_DIMENSIONS]
    for _, r in sample.iterrows():
        base = {"ref": r[ref_col], "photo_urls_or_paths": r[image_col],
                "is_double_coded": r[ref_col] in double_coded_refs,
                "rater": "rater_1"}
        base.update({f"{d}_0_or_1": "" for d in dims})
        base["notes"] = ""
        rows.append(base)
        if r[ref_col] in double_coded_refs:
            second = dict(base); second["rater"] = "rater_2"
            rows.append(second)
    t = pd.DataFrame(rows)
    t.to_csv(out_path, index=False)
    print(f"Wrote manual audit template ({len(sample)} listings, "
          f"{len(double_coded_refs)} double-coded) to {out_path}")
    return t


def compute_inter_rater_reliability(completed_csv):
    from sklearn.metrics import cohen_kappa_score
    df = pd.read_csv(completed_csv)
    double = df[df["is_double_coded"] == True]
    dims = [c[:-len("_0_or_1")] for c in df.columns if c.endswith("_0_or_1")]
    results = []
    for dim in dims:
        col = f"{dim}_0_or_1"
        pivot = double.pivot_table(index="ref", columns="rater", values=col, aggfunc="first").dropna()
        if len(pivot) < 2 or "rater_1" not in pivot or "rater_2" not in pivot:
            continue
        kappa = cohen_kappa_score(pivot["rater_1"], pivot["rater_2"])
        agree = (pivot["rater_1"] == pivot["rater_2"]).mean() * 100
        results.append({"dimension": dim, "cohen_kappa": kappa,
                         "raw_agreement_pct": agree, "n_double_coded": len(pivot)})
    out = pd.DataFrame(results)
    print(out.to_string(index=False))
    return out


def parse_photo_urls_column(series):
    """JSON-array OR '; '-separated photo URL lists. The scraper's CSV
    writes JSON arrays (e.g. '["url1", "url2"]'); older scrapes used
    '; ' joining; this handles both without failing on either."""
    def parse_one(s):
        if pd.isna(s):
            return []
        s = str(s).strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                arr = json.loads(s)
                return [str(u).strip() for u in arr if str(u).strip()]
            except (json.JSONDecodeError, TypeError):
                pass
        return [u.strip() for u in s.split(";") if u.strip()]
    return series.apply(parse_one)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path, default=Path("/content/cleaned_listings.csv"))
    p.add_argument("--output", type=Path, default=Path("/content/image_features.csv"))
    p.add_argument("--ref-col",   type=str, default="ref")
    p.add_argument("--image-col", type=str, default="photo_urls")
    p.add_argument("--sample-size", type=int, default=150)
    p.add_argument("--max-images-per-listing", type=int, default=6)
    p.add_argument("--mode", choices=["clip", "manual-template", "manual-reliability"],
                   default="clip")
    p.add_argument("--completed-manual-csv", type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["photo_urls_list"] = parse_photo_urls_column(df[args.image_col])
    has_photos = df["photo_urls_list"].apply(len) > 0
    n_with = int(has_photos.sum())
    print(f"{n_with} of {len(df)} listings have >=1 usable photo URL")
    if n_with < args.sample_size:
        print(f"WARNING: fewer than --sample-size={args.sample_size} listings "
              f"have photos; the brief requires >=150. State this explicitly "
              f"in your write-up if you cannot meet it.")

    if args.mode == "manual-reliability":
        if not args.completed_manual_csv:
            raise SystemExit("--completed-manual-csv is required")
        compute_inter_rater_reliability(args.completed_manual_csv)
        return

    if args.mode == "manual-template":
        out = args.output.parent / "manual_visual_audit_template.csv"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        generate_manual_audit_template(df[has_photos], args.ref_col,
                                        args.image_col, args.sample_size, out)
        return

    sample = (df[has_photos].sample(n=min(args.sample_size, n_with), random_state=42)
              if n_with > args.sample_size else df[has_photos])
    image_urls_by_ref = dict(zip(sample[args.ref_col], sample["photo_urls_list"]))

    t0 = time.time()
    scores = score_images_with_clip(image_urls_by_ref,
                                     max_images_per_listing=args.max_images_per_listing)
    print(f"CLIP scoring took {time.time() - t0:.0f}s for {len(scores)} listings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"Wrote image features to {args.output}")


if __name__ == "__main__":
    main()

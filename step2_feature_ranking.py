#!/usr/bin/env python3
"""

"""
import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

# ---- Default output directory ----
DEFAULT_OUTDIR = Path("data/features_ranking")
DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def main(features_csv: str, labels_csv: str, outdir: str, n_estimators: int, max_depth: int, importance_threshold: float):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load inputs
    X = pd.read_csv(features_csv)
    y_df = pd.read_csv(labels_csv)

    # Basic sanity checks
    if X.shape[0] != y_df.shape[0]:
        raise SystemExit(f"ERROR: number of rows mismatch: features={X.shape[0]} rows, labels={y_df.shape[0]} rows")

    # If labels file has header, take the first column as label
    label_col = y_df.columns[0]
    y = y_df[label_col].astype(str).copy()

    print(f"Loaded features: rows={X.shape[0]:,}, cols={X.shape[1]:,}")
    print(f"Loaded labels: column='{label_col}', unique_labels={y.nunique()}")

    # 2) Encode labels to integers (save mapping)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_mapping = {int(v): str(k) for v, k in enumerate(le.classes_)}
    save_json(label_mapping, outdir / "label_encoder.json")
    print(f"Saved label mapping -> {outdir / 'label_encoder.json'}")

    # 3) Convert X to numeric numpy array (should already be numeric from Step1)
    #    Keep column order stable.
    feature_names = list(X.columns)
    X_values = X.values
    if not np.issubdtype(X_values.dtype, np.number):
        # attempt to coerce
        X_values = X.astype(float).values

    # 4) Fit ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    clf.fit(X_values, y_enc)
    print("Fitted ExtraTreesClassifier")

    # Save model
    joblib.dump(clf, outdir / "extratrees_model.joblib")
    print(f"Saved ExtraTrees model -> {outdir / 'extratrees_model.joblib'}")

    # 5) Extract feature importances and produce sorted DataFrame
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["rank"] = fi_df.index + 1
    fi_csv = outdir / "feature_importance.csv"
    fi_df.to_csv(fi_csv, index=False)
    print(f"Saved feature importances -> {fi_csv}")

    # 6) Select features by threshold (paper used > 0.001). Fallback to top-61 if threshold selects too few.
    selected = fi_df[fi_df["importance"] > importance_threshold]["feature"].tolist()
    if len(selected) < 10:
        # fallback: select top-61 (paper ~61)
        fallback_k = min(61, len(feature_names))
        selected = fi_df.head(fallback_k)["feature"].tolist()
        print(f"Threshold {importance_threshold} selected {len(selected)} features (<10). Falling back to top-{fallback_k} selection.")
    else:
        print(f"Selected {len(selected)} features using threshold > {importance_threshold}")

    # 7) Save selected features and full ordered list
    sel_txt = outdir / "selected_features.txt"
    with open(sel_txt, "w", encoding="utf-8") as fh:
        for f in selected:
            fh.write(f + "\n")
    print(f"Saved selected features -> {sel_txt}")

    ordered_json = outdir / "feature_order.json"
    save_json(fi_df["feature"].tolist(), ordered_json)
    print(f"Saved ordered feature list -> {ordered_json}")

    # 8) Plot importances (top 30 or all if fewer)
    top_k = min(30, len(fi_df))
    fig, ax = plt.subplots(figsize=(10, max(4, top_k*0.25)))
    ax.barh(fi_df.head(top_k)["feature"][::-1], fi_df.head(top_k)["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances (ExtraTrees)")
    plt.tight_layout()
    plot_path = outdir / "feature_importance.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved importance plot -> {plot_path}")

    # 9) Save the importance numbers also to a small JSON summary
    summary = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_selected": int(len(selected)),
        "selection_method": f"importance>{importance_threshold} (fallback top-61)",
        "top_features": fi_df.head(10)["feature"].tolist()
    }
    save_json(summary, outdir / "summary.json")
    print(f"Saved summary -> {outdir / 'summary.json'}")

    # Print concise summary to stdout
    print("-" * 60)
    print(f"Rows: {X.shape[0]:,}   Features: {X.shape[1]:,}")
    print(f"Selected features: {len(selected)}  (saved to {sel_txt})")
    print("Top 10 features (by importance):")
    for i, row in fi_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:40s} importance={row['importance']:.6f}")
    print("-" * 60)

    return {
        "feature_importance_csv": str(fi_csv),
        "selected_features_txt": str(sel_txt),
        "feature_order_json": str(ordered_json),
        "model_path": str(outdir / "extratrees_model.joblib"),
        "plot_path": str(plot_path),
        "summary": summary
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Feature ranking & selection using ExtraTrees.")
    parser.add_argument("--features", "-f", required=True, help="Path to features CSV (from Step 1).")
    parser.add_argument("--labels", "-l", required=True, help="Path to labels CSV (from Step 1).")
    parser.add_argument("--outdir", "-o", default=str(DEFAULT_OUTDIR), help="Output directory (default: data/features_ranking).")
    parser.add_argument("--n_estimators", type=int, default=250, help="Number of trees for ExtraTrees (default: 250).")
    parser.add_argument("--max_depth", type=int, default=16, help="Max depth for ExtraTrees (default: 16).")
    parser.add_argument("--importance_threshold", type=float, default=0.001, help="Importance threshold to select features (default: 0.001).")
    args = parser.parse_args()
    main(args.features, args.labels, args.outdir, args.n_estimators, args.max_depth, args.importance_threshold)

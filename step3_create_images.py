import argparse
import json
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- defaults ---
DEFAULT_OUT_PRE = Path("data/preprocessing")
DEFAULT_OUT_IMAGES = Path("data/images")
DEFAULT_VALIDATION = Path("data/validation")
DEFAULT_LABEL_ENCODER = Path("data/features_ranking/label_encoder.json")  # try to reuse if present

DEFAULT_OUT_PRE.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT_IMAGES.mkdir(parents=True, exist_ok=True)
DEFAULT_VALIDATION.mkdir(parents=True, exist_ok=True)


def load_selected(selected_path: Path):
    if not selected_path.exists():
        raise SystemExit(f"ERROR: selected features file not found: {selected_path}")
    with open(selected_path, "r", encoding="utf-8") as fh:
        features = [line.strip() for line in fh if line.strip()]
    return features


def try_load_label_mapping(path: Path):
    """If label mapping exists (from Step2), load and invert so we get label->int mapping.
       Expected format from Step2: { "0": "CLASSNAME", "1": "OTHER" }
       We will invert to { "CLASSNAME": 0, ... }.
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # invert if possible
    try:
        # raw keys may be strings of ints; values are class names
        mapping = {v: int(k) for k, v in raw.items()}
        return mapping
    except Exception:
        # if raw is label->int already, just return it
        if all(isinstance(v, int) for v in raw.values()):
            return raw
        return None


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def make_images_from_array(X_scaled, feature_indices):
    """
    X_scaled: numpy array shape (N, n_selected) with values already scaled to [0,255]
    feature_indices: list of cell indices where each feature should be placed (len = n_selected)
    returns: images numpy array shape (N, 8, 8, 1)
    """
    N = X_scaled.shape[0]
    images = np.zeros((N, 8, 8), dtype=np.float32)
    for col_idx, cell_idx in enumerate(feature_indices):
        # place column values into flat cell index
        r = cell_idx // 8
        c = cell_idx % 8
        images[:, r, c] = X_scaled[:, col_idx]
    images = images.reshape((N, 8, 8, 1))
    return images


def plot_sample_grid(X_images, y, outpath: Path, classes_map: dict = None, n=16):
    """Save a small grid of sample images for quick visual check (n must be square)."""
    n = min(n, X_images.shape[0])
    per_row = int(np.sqrt(n))
    fig, axs = plt.subplots(per_row, per_row, figsize=(per_row * 1.5, per_row * 1.5))
    idxs = np.linspace(0, X_images.shape[0]-1, n, dtype=int)
    for i, ax in enumerate(axs.flat):
        im = X_images[idxs[i]].squeeze()
        ax.imshow(im, interpolation="nearest")
        if classes_map is not None:
            ax.set_title(str(classes_map.get(int(y[idxs[i]]), y[idxs[i]])), fontsize=6)
        else:
            ax.set_title(str(int(y[idxs[i]])), fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main(features_csv, labels_csv, selected_path, out_pre, out_images, out_validation, label_encoder_path):
    # parse paths
    features_csv = Path(features_csv)
    labels_csv = Path(labels_csv)
    selected_path = Path(selected_path)
    out_pre = Path(out_pre)
    out_images = Path(out_images)
    out_validation = Path(out_validation)

    # load selected features
    selected_features = load_selected(selected_path)
    n_selected = len(selected_features)
    if n_selected <= 0 or n_selected > 64:
        raise SystemExit(f"ERROR: number of selected features must be 1..64. Found: {n_selected}")

    # load features and labels
    if not features_csv.exists() or not labels_csv.exists():
        raise SystemExit("ERROR: features or labels CSV not found.")
    X_all = pd.read_csv(features_csv)
    y_df = pd.read_csv(labels_csv)
    if X_all.shape[0] != y_df.shape[0]:
        raise SystemExit("ERROR: features and labels row count mismatch.")

    # ensure all selected features exist in X_all
    missing = [f for f in selected_features if f not in X_all.columns]
    if missing:
        raise SystemExit(f"ERROR: the following selected features are missing from features CSV: {missing}")

    # extract selected features matrix
    X_sel = X_all[selected_features].copy()

    # coerce numeric and impute if necessary (median)
    for c in X_sel.columns:
        if not pd.api.types.is_numeric_dtype(X_sel[c]):
            X_sel[c] = pd.to_numeric(X_sel[c], errors="coerce")
    # replace inf
    X_sel = X_sel.replace([np.inf, -np.inf], np.nan)
    # impute NaN with median (defensive)
    for c in X_sel.columns:
        if X_sel[c].isnull().any():
            med = X_sel[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            X_sel[c] = X_sel[c].fillna(med)

    # prepare labels (load or create mapping)
    # try to reuse mapping from step2 if available
    existing_map = try_load_label_mapping(Path(label_encoder_path))
    label_series = y_df.iloc[:, 0].astype(str).copy()
    if existing_map is not None:
        # invert existing_map to label -> int
        label_to_int = {label: code for label, code in existing_map.items()} if all(isinstance(k, str) for k in existing_map.keys()) else {v: int(k) for k, v in existing_map.items()}
        # Ensure all labels are in mapping; otherwise fallback to LabelEncoder
        if set(label_series.unique()) <= set(label_to_int.keys()):
            y_enc = label_series.map(label_to_int).astype(int).values
        else:
            # fallback: create fresh encoder
            le = LabelEncoder()
            y_enc = le.fit_transform(label_series)
            label_to_int = {str(cl): int(i) for i, cl in enumerate(le.classes_)}
            save_json(label_to_int, out_pre / "label_encoder.json")
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(label_series)
        label_to_int = {str(cl): int(i) for i, cl in enumerate(le.classes_)}
        save_json(label_to_int, out_pre / "label_encoder.json")

    # Save label mapping (ensure it's available to Person 2)
    save_json(label_to_int, out_pre / "label_encoder.json")

    # --- scaling: MinMax to [0,255] ---
    scaler = MinMaxScaler(feature_range=(0, 255))
    X_values = X_sel.values.astype(np.float32)
    scaler.fit(X_values)
    X_scaled = scaler.transform(X_values)
    # clip for safety
    X_scaled = np.clip(X_scaled, 0.0, 255.0)

    # save scaler and params
    joblib.dump(scaler, out_pre / "scaler.joblib")
    # save min/max params for reproducibility
    scaler_params = {
        "feature_names": selected_features,
        "data_min": [float(x) for x in scaler.data_min_.tolist()],
        "data_max": [float(x) for x in scaler.data_max_.tolist()],
        "data_range": [float(x) for x in scaler.data_range_.tolist()],
        "feature_range": [0.0, 255.0]
    }
    save_json(scaler_params, out_pre / "scaler_params.json")

    # --- create mapping: feature -> cell index (row-major) ---
    # follow paper: put features into first n positions in the 8x8 grid (row-major)
    feature_to_cell = {}
    for idx, feat in enumerate(selected_features):
        feature_to_cell[feat] = idx  # 0..n_selected-1
    unused_cells = list(range(n_selected, 64))
    mapping = {
        "feature_to_cell_index": feature_to_cell,
        "n_selected": n_selected,
        "grid_shape": [8, 8],
        "unused_cells": unused_cells,
        "ordering": "row_major (0->(0,0),1->(0,1),...)"
    }
    save_json(mapping, out_pre / "feature_to_cell_map.json")

    # --- build images ---
    feature_indices = [feature_to_cell[feat] for feat in selected_features]
    X_images = make_images_from_array(X_scaled, feature_indices)  # shape (N,8,8,1), float32

    # --- train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_images, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # save arrays
    np.save(out_images / "X_train.npy", X_train)
    np.save(out_images / "y_train.npy", y_train)
    np.save(out_images / "X_test.npy", X_test)
    np.save(out_images / "y_test.npy", y_test)

    # --- validation report ---
    report = {
        "n_rows_total": int(X_images.shape[0]),
        "n_selected_features": int(n_selected),
        "grid_shape": [8, 8, 1],
        "X_train_shape": [int(x) for x in X_train.shape],
        "X_test_shape": [int(x) for x in X_test.shape],
        "y_train_shape": [int(x) for x in y_train.shape],
        "y_test_shape": [int(x) for x in y_test.shape],
        "label_mapping": label_to_int,
        "label_distribution_total": {str(k): int(v) for k, v in zip(*np.unique(y_enc, return_counts=True))},
        "label_distribution_train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "label_distribution_test": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "scaler_saved": str(out_pre / "scaler.joblib"),
        "feature_to_cell_map": str(out_pre / "feature_to_cell_map.json"),
        "note": "Values scaled to [0,255]. Unused grid cells set to 0."
    }
    save_json(report, out_validation / "validation_report.json")

    # Also save a human-readable txt
    with open(out_validation / "validation_report.txt", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(report, indent=2))

    # Save a small sample image grid for quick sanity check (optional but useful)
    try:
        plot_sample_grid(X_train, y_train, out_validation / "sample_images.png", classes_map={v: k for k, v in label_to_int.items()}, n=16)
    except Exception:
        # not fatal
        pass

    # print concise summary
    print(f"Total rows processed: {X_images.shape[0]}")
    print(f"Selected features: {n_selected}")
    print(f"Saved scaler -> {out_pre / 'scaler.joblib'}")
    print(f"Saved feature->cell map -> {out_pre / 'feature_to_cell_map.json'}")
    print(f"Saved images: X_train.npy ({X_train.shape}), X_test.npy ({X_test.shape})")
    print(f"Saved labels: y_train.npy ({y_train.shape}), y_test.npy ({y_test.shape})")
    print(f"Saved validation report -> {out_validation / 'validation_report.txt'}")
    print("Step 3 complete. Give the files in data/images and data/preprocessing to Person 2 for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Normalize selected features and create 8x8 images.")
    parser.add_argument("--features", "-f", required=True, help="Path to features CSV (from Step 1).")
    parser.add_argument("--labels", "-l", required=True, help="Path to labels CSV (from Step 1).")
    parser.add_argument("--selected", "-s", required=True, help="Path to selected_features.txt (from Step 2).")
    parser.add_argument("--out_pre", default=str(DEFAULT_OUT_PRE), help="Output preprocessing directory (default: data/preprocessing).")
    parser.add_argument("--out_images", default=str(DEFAULT_OUT_IMAGES), help="Output images directory (default: data/images).")
    parser.add_argument("--out_validation", default=str(DEFAULT_VALIDATION), help="Validation output directory (default: data/validation).")
    parser.add_argument("--label_encoder", default=str(DEFAULT_LABEL_ENCODER), help="Optional existing label encoder JSON (from Step2).")
    args = parser.parse_args()

    main(args.features, args.labels, args.selected, args.out_pre, args.out_images, args.out_validation, args.label_encoder)

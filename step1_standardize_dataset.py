#!/usr/bin/env python3
"""
s
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----- configurable output paths -----
OUT_DIR = Path("data/clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_colname(c: str) -> str:
    """Normalize column name for safe matching (lowercase, strip, collapse spaces/dots)."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in c.strip())


def _find_label_columns(cols):
    """Return a tuple: (label_col_to_keep, other_label_cols_list) or (None, []) if none found."""
    normalized = {c: _normalize_colname(c) for c in cols}
    # Common normalized patterns for label and duplicate
    label_candidates = [c for c, nc in normalized.items() if nc == "label" or nc.startswith("label_")]
    # Prefer exact "label"
    if any(_normalize_colname(c) == "label" for c in cols):
        keep = next(c for c in cols if _normalize_colname(c) == "label")
        others = [c for c in label_candidates if c != keep]
        return keep, others
    # else if only variants like label_1 exist, pick the first as keep
    if label_candidates:
        keep = label_candidates[0]
        others = [c for c in label_candidates if c != keep]
        return keep, others
    return None, []


def _detect_identifier_columns(cols):
    """Detect likely identifier columns to drop (flow id, src/dst ip, src/dst port, timestamp)"""
    normalized = {c: _normalize_colname(c) for c in cols}
    identifiers = []
    id_patterns = {
        "flow_id": {"flowid", "flow_id", "flow", "flowid"},
        "src_ip": {"srcip", "sourceip", "src_ip", "source_ip", "src_ip_address"},
        "dst_ip": {"dstip", "destinationip", "dst_ip", "destination_ip", "dst_ip_address"},
        "src_port": {"srcport", "sourceport", "src_port"},
        "dst_port": {"dstport", "destinationport", "dst_port"},
        "timestamp": {"timestamp", "time", "starttime", "lasttime"},
    }
    for orig, n in normalized.items():
        for key, patset in id_patterns.items():
            if n in patset:
                identifiers.append(orig)
                break
    # Be conservative: only drop if name matches exactly one of the expected tokens above.
    return identifiers


def main(input_csv: str):
    # --- 1) load CSV safely ---
    if not os.path.isfile(input_csv):
        raise SystemExit(f"ERROR: input file not found: {input_csv}")

    try:
        df = pd.read_csv(input_csv, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        # fallback encoding
        df = pd.read_csv(input_csv, low_memory=False, encoding="latin-1")

    print(f"Loaded '{input_csv}' rows={len(df):,} cols={len(df.columns):,}")

    # standardize column name list (preserve originals)
    original_cols = list(df.columns)

    # --- 2) detect label columns ---
    label_keep, label_others = _find_label_columns(original_cols)
    if label_keep is None:
        raise SystemExit("ERROR: no label column found. Expected a column named 'Label' or variant. "
                         "Please check the CSV header.")
    print(f"Label chosen: '{label_keep}'")
    if label_others:
        print(f"Other label-like columns detected and will be dropped: {label_others}")

    # --- 3) detect and drop identifier columns if present ---
    id_cols = _detect_identifier_columns(original_cols)
    if id_cols:
        print(f"Identifier-like columns detected and will be dropped if present: {id_cols}")
        df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")
    else:
        print("No identifier-like columns detected (good).")

    # --- 4) drop duplicate label columns (keep only chosen label) ---
    to_drop = [c for c in label_others if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    # ensure only one label column exists now
    if label_keep not in df.columns:
        # if chosen label was not present (edge-case), pick any remaining label-like col
        remaining_labels = [c for c in df.columns if _normalize_colname(c).startswith("label")]
        if remaining_labels:
            label_keep = remaining_labels[0]
            print(f"Warning: original chosen label not present; using '{label_keep}' instead.")
        else:
            raise SystemExit("ERROR: no label column remaining after cleanup.")

    # --- 5) separate features and label ---
    label_series = df[label_keep].copy()
    features_df = df.drop(columns=[label_keep])

    # --- 6) convert protocol if necessary (factorize non-numeric protocol) ---
    proto_col = None
    for c in features_df.columns:
        if _normalize_colname(c) == "protocol":
            proto_col = c
            break

    protocol_map = {}
    if proto_col is not None:
        # if protocol is already numeric, leave it; otherwise factorize
        if not pd.api.types.is_numeric_dtype(features_df[proto_col]):
            proto_vals = features_df[proto_col].astype(str).fillna("NA")
            codes, uniques = pd.factorize(proto_vals)
            features_df[proto_col] = codes  # integer codes starting at 0
            protocol_map = {str(u): int(i) for i, u in enumerate(uniques)}
            print(f"Protocol column '{proto_col}' factorized to numeric. Mapping size={len(protocol_map)}")
        else:
            print(f"Protocol column '{proto_col}' already numeric. No factorization needed.")
    else:
        print("Protocol column not found among features (proceeding).")

    # --- 7) convert all feature columns to numeric where feasible (coerce errors -> NaN) ---
    for c in features_df.columns:
        # skip protocol because already handled
        if c == proto_col:
            continue
        # attempt numeric conversion
        if not pd.api.types.is_numeric_dtype(features_df[c]):
            features_df[c] = pd.to_numeric(features_df[c], errors="coerce")

    # --- 8) replace inf/-inf with NaN and impute NaNs with median for numeric columns ---
    num_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    # replace ±inf
    features_df[num_cols] = features_df[num_cols].replace([np.inf, -np.inf], np.nan)

    # compute medians (for imputation) and fill
    medians = {}
    for c in num_cols:
        med = features_df[c].median(skipna=True)
        if pd.isna(med):
            # if entire column is NaN (degenerate), fill with 0
            med = 0.0
        medians[c] = float(med)
        features_df[c] = features_df[c].fillna(med)

    # For any non-numeric columns left (rare), fill na with empty string
    non_num_cols = [c for c in features_df.columns if c not in num_cols]
    for c in non_num_cols:
        features_df[c] = features_df[c].fillna("")

    # --- 9) drop exact duplicate rows (features + label) ---
    before = len(features_df)
    combined = pd.concat([features_df, label_series.rename("_LABEL_TEMP_")], axis=1)
    combined = combined.drop_duplicates(keep="first")
    after = len(combined)
    if after < before:
        print(f"Dropped {before - after:,} exact duplicate rows.")
    # split back
    label_series = combined["_LABEL_TEMP_"].copy()
    features_df = combined.drop(columns=["_LABEL_TEMP_"])

    # --- 10) final sanity checks ---
    if features_df.isnull().any().any():
        print("Warning: some NaNs remain in features after imputation (unexpected).")
    if label_series.isnull().any():
        n_missing_labels = int(label_series.isnull().sum())
        print(f"Warning: {n_missing_labels} rows have missing labels and will be dropped.")
        mask = label_series.notnull()
        features_df = features_df.loc[mask].reset_index(drop=True)
        label_series = label_series.loc[mask].reset_index(drop=True)
    else:
        features_df = features_df.reset_index(drop=True)
        label_series = label_series.reset_index(drop=True)

    # --- 11) save outputs ---
    cleaned_path = OUT_DIR / "cleaned_flows.csv"
    features_path = OUT_DIR / "features.csv"
    labels_path = OUT_DIR / "labels.csv"
    feature_list_path = OUT_DIR / "feature_list.txt"
    protocol_map_path = OUT_DIR / "protocol_map.json"

    # Save cleaned dataframe (features + label with the original label column name)
    cleaned_df = pd.concat([features_df, label_series.rename(label_keep)], axis=1)
    cleaned_df.to_csv(cleaned_path, index=False)
    features_df.to_csv(features_path, index=False)
    label_series.to_frame(name=label_keep).to_csv(labels_path, index=False)

    # Save feature list
    with open(feature_list_path, "w", encoding="utf-8") as fh:
        for c in features_df.columns:
            fh.write(f"{c}\n")

    # Save protocol mapping if created
    if protocol_map:
        with open(protocol_map_path, "w", encoding="utf-8") as fh:
            json.dump(protocol_map, fh, indent=2)
    else:
        # ensure empty file exists
        with open(protocol_map_path, "w", encoding="utf-8") as fh:
            json.dump({}, fh)

    # print summary
    n_rows, n_cols = cleaned_df.shape
    print("-" * 60)
    print(f"Saved cleaned dataframe -> {cleaned_path}  (rows={n_rows:,}, cols={n_cols:,})")
    print(f"Saved features only        -> {features_path}  (cols={len(features_df.columns):,})")
    print(f"Saved labels only          -> {labels_path}")
    print(f"Saved feature list         -> {feature_list_path}")
    print(f"Saved protocol mapping     -> {protocol_map_path}")
    print("-" * 60)
    print("Top 10 features (by column order):", list(features_df.columns[:10]))
    print("Label value counts:")
    print(label_series.value_counts(dropna=False).to_string())
    print("Step 1 complete.")

    return {
        "cleaned_path": str(cleaned_path),
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "feature_list_path": str(feature_list_path),
        "protocol_map_path": str(protocol_map_path),
        "n_rows": n_rows,
        "n_features": len(features_df.columns),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize CICDarknet-improved CSV (Step 1).")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file (raw dataset).")
    args = parser.parse_args()
    main(args.input)

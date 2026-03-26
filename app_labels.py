# step5_prepare_app_labels.py

import argparse
import pandas as pd
import json
import os
import sys

APP_CLASSES = {
    "BROWSING": 0,
    "CHAT": 1,
    "EMAIL": 2,
    "FILE-TRANSFER": 3,
    "AUDIO-STREAMING": 4,
    "VIDEO-STREAMING": 5,
    "VOIP": 6,
    "P2P": 7
}

def normalize_label(label: str):
    label = label.strip().upper()
    if label in APP_CLASSES:
        return label
    if "FILE" in label:
        return "FILE-TRANSFER"
    return None

def main(args):
    out_dir = "data/stage2_app/labels"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.labels)

    if "Label.1" not in df.columns:
        sys.exit("ERROR: 'Label.1' column not found")

    app_names = df["Label.1"].apply(normalize_label)

    if app_names.isnull().any():
        bad = df.loc[app_names.isnull(), "Label.1"].unique()
        sys.exit(f"ERROR: Unknown application labels: {bad}")

    app_ids = app_names.map(APP_CLASSES)

    out_df = pd.DataFrame({
        "app_name": app_names,
        "app_label": app_ids
    })

    out_df.to_csv(f"{out_dir}/app_labels.csv", index=False)

    with open(f"{out_dir}/app_label_map.json", "w") as f:
        json.dump(APP_CLASSES, f, indent=2)

    dist = out_df["app_name"].value_counts()
    with open(f"{out_dir}/app_label_distribution.txt", "w") as f:
        f.write(dist.to_string())

    print("Stage-2 Application Labeling COMPLETE")
    print("Samples:", len(out_df))
    print(dist)
    print("Saved:")
    print(" -> data/stage2_app/labels/app_labels.csv")
    print(" -> data/stage2_app/labels/app_label_map.json")
    print(" -> data/stage2_app/labels/app_label_distribution.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", "-l", required=True)
    args = parser.parse_args()
    main(args)

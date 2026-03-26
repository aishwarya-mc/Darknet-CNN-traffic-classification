# step4_filter_darknet_for_app.py

import argparse
import json
import os
import pandas as pd


DARKNET_LABELS = {"Tor", "VPN"}


def parse_args():
    parser = argparse.ArgumentParser(description="Filter darknet traffic (Tor + VPN) for Stage-2 application classification")
    parser.add_argument("--features", "-f", required=True, help="Path to features.csv")
    parser.add_argument("--labels", "-l", required=True, help="Path to labels.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    # Output directory
    out_dir = os.path.join("data", "stage2_app", "filtered")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    X = pd.read_csv(args.features)
    y = pd.read_csv(args.labels)

    if "Label" not in y.columns:
        raise ValueError("labels.csv must contain a 'Label' column")

    if len(X) != len(y):
        raise ValueError("Features and labels row count mismatch")

    # Combine for safe filtering
    df = X.copy()
    df["Label"] = y["Label"]

    # Filter darknet only
    df_darknet = df[df["Label"].isin(DARKNET_LABELS)].reset_index(drop=True)

    if df_darknet.empty:
        raise RuntimeError("No darknet samples found (Tor/VPN). Check labels.")

    # Split back
    X_darknet = df_darknet.drop(columns=["Label"])
    y_darknet = df_darknet[["Label"]]

    # Save outputs
    feat_path = os.path.join(out_dir, "darknet_features.csv")
    label_path = os.path.join(out_dir, "darknet_labels.csv")

    X_darknet.to_csv(feat_path, index=False)
    y_darknet.to_csv(label_path, index=False)

    # Summary
    summary = {
        "total_samples": len(df_darknet),
        "label_distribution": y_darknet["Label"].value_counts().to_dict(),
        "num_features": X_darknet.shape[1],
        "labels_used": sorted(DARKNET_LABELS)
    }

    with open(os.path.join(out_dir, "darknet_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("------------------------------------------------------------")
    print("Stage-2 Darknet Filtering Complete")
    print(f"Total darknet samples: {len(df_darknet)}")
    print("Label distribution:")
    for k, v in summary["label_distribution"].items():
        print(f"  {k}: {v}")
    print("Saved files:")
    print(f"  -> {feat_path}")
    print(f"  -> {label_path}")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()

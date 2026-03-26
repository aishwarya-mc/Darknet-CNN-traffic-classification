import pandas as pd
import json
import os
import sys


APP_MAPPING = {
    "Browsing": 0,
    "Chat": 1,
    "Email": 2,
    "File-Transfer": 3,
    "Audio-Streaming": 4,
    "Video-Streaming": 5,
    "VOIP": 6,
    "P2P": 7
}


def main():

    input_path = "data/stage2_app/filtered/darknet_full.csv"

    if not os.path.exists(input_path):
        sys.exit("ERROR: darknet_full.csv not found")

    df = pd.read_csv(input_path)

    if "Label.1" not in df.columns:
        sys.exit("ERROR: 'Label.1' column missing")

    df["app_label"] = df["Label.1"].map(APP_MAPPING)

    if df["app_label"].isnull().any():
        bad = df[df["app_label"].isnull()]["Label.1"].unique()
        sys.exit(f"ERROR: Unknown labels found -> {bad}")

    output_dir = "data/stage2_app/labels"
    os.makedirs(output_dir, exist_ok=True)

    df[["app_label"]].to_csv(
        os.path.join(output_dir, "app_labels_numeric.csv"),
        index=False
    )

    with open(os.path.join(output_dir, "app_label_map.json"), "w") as f:
        json.dump(APP_MAPPING, f, indent=2)

    print("Stage-2 label encoding COMPLETE")
    print("Total samples:", len(df))
    print("\nClass distribution:")
    print(df["Label.1"].value_counts())

    print("\nSaved:")
    print(" -> data/stage2_app/labels/app_labels_numeric.csv")
    print(" -> data/stage2_app/labels/app_label_map.json")


if __name__ == "__main__":
    main()

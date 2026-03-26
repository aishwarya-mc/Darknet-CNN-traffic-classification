import argparse
import pandas as pd
import os
import sys


def main(input_path):

    if not os.path.exists(input_path):
        sys.exit(f"ERROR: File not found -> {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {"Label", "Label.1"}
    if not required_cols.issubset(df.columns):
        sys.exit("ERROR: Dataset must contain 'Label' and 'Label.1' columns")

    print(f"Loaded dataset: {len(df):,} rows")

    # Keep only Tor and VPN traffic
    darknet_df = df[df["Label"].isin(["Tor", "VPN"])].copy()

    if darknet_df.empty:
        sys.exit("ERROR: No Tor/VPN samples found")

    before = len(darknet_df)
    darknet_df.drop_duplicates(inplace=True)
    after = len(darknet_df)

    print(f"Darknet samples kept: {after:,}")
    print(f"Duplicates removed: {before - after:,}")

    # Remove rows with missing application label
    darknet_df = darknet_df[darknet_df["Label.1"].notna()]

    # Create output directory
    output_dir = "data/stage2_app/filtered"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "darknet_full.csv")
    darknet_df.to_csv(output_path, index=False)

    print("\nSaved file:")
    print(f" -> {output_path}")

    print("\nApplication distribution:")
    print(darknet_df["Label.1"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True,
                        help="Path to original output.csv")
    args = parser.parse_args()

    main(args.input)

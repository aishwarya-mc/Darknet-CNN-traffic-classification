import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def main():

    input_path = "data/stage2_app/filtered/darknet_full.csv"
    labels_path = "data/stage2_app/labels/app_labels_numeric.csv"
    selected_features_path = "data/features_ranking/selected_features.txt"

    if not os.path.exists(input_path):
        sys.exit("ERROR: darknet_full.csv not found")

    if not os.path.exists(labels_path):
        sys.exit("ERROR: app_labels_numeric.csv not found")

    if not os.path.exists(selected_features_path):
        sys.exit("ERROR: selected_features.txt not found")

    df = pd.read_csv(input_path)
    labels = pd.read_csv(labels_path)

    with open(selected_features_path, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing features in dataset -> {missing}")

    X = df[selected_features].values
    y = labels["app_label"].values

    scaler = MinMaxScaler(feature_range=(0, 255))
    X_scaled = scaler.fit_transform(X)

    # Create 8x8 images (61 features placed, rest zero)
    num_samples = X_scaled.shape[0]
    images = np.zeros((num_samples, 8, 8, 1), dtype=np.float32)

    for i in range(num_samples):
        flat = X_scaled[i]
        image = np.zeros((64,))
        image[:61] = flat[:61]
        images[i, :, :, 0] = image.reshape(8, 8)

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    output_dir = "data/stage2_app/images"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print("Stage-2 Image Creation COMPLETE")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nSaved:")
    print(" -> data/stage2_app/images/X_train.npy")
    print(" -> data/stage2_app/images/X_test.npy")
    print(" -> data/stage2_app/images/y_train.npy")
    print(" -> data/stage2_app/images/y_test.npy")


if __name__ == "__main__":
    main()

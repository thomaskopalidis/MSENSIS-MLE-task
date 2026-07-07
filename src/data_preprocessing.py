import os 
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["cats", "dogs"]

def build_manifest(raw_dir: str) -> pd.DataFrame:
    """
    Build a manifest DataFrame from the raw data directory.

    Args:
        raw_dir (str): Path to the raw data directory.

    Returns:
        pd.DataFrame: Manifest DataFrame containing information about the images.
    """
    records = []
    for label in CLASSES:
        folder = Path(raw_dir) / label
        for img_path in folder.glob("*.*"):
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify that the image is not corrupted
                records.append({"path": str(img_path), "label": label})
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return pd.DataFrame(records)    

def stratified_split(
    df: pd.DataFrame, 
    train_size: float = 0.8, 
    val_size: float = 0.1, 
    test_size: float = 0.1, 
    random_state: int = 42,
):
    """
    Split a manifest DataFrame into stratified train, validation, and test sets.
    preserving class balance across all three splits.

    Returns:
        tuple: (train_df, val_df, test_df) 
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=df["label"], random_state=random_state
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val, stratify=temp_df["label"], random_state=random_state
    )
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir: str = "data/processed"):

    """
    Save the train, validation, and test DataFrames to CSV files.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        output_dir (str): Directory to save the CSV files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    val_df.to_csv(Path(output_dir) / "val.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)


def main():
    RAW_DIR = "data/raw"
    OUTPUT_DIR = "data/processed"

    print("Building manifest...")
    df = build_manifest(RAW_DIR)

    print(f"Valid images: {len(df)}")
    print("Class balance: \n", df["label"].value_counts())

    print("\Splitting into train/val/test (80/10/10, stratified)...")
    train_df, val_df, test_df = stratified_split(df)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    save_splits(train_df, val_df, test_df, OUTPUT_DIR)
    print(f"\nSaved splits to {OUTPUT_DIR}/")    

if __name__ == "__main__":
    main()
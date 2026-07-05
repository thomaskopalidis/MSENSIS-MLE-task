import os 
from pathlib import Path
from PIL import Image
import pandas as pd

def build_manifest(raw_dir: str) -> pd.DataFrame:
    """
    Build a manifest DataFrame from the raw data directory.

    Args:
        raw_dir (str): Path to the raw data directory.

    Returns:
        pd.DataFrame: Manifest DataFrame containing information about the images.
    """
    records = []
    for label in ["cats", "dogs"]:
        folder = Path(raw_dir) / label
        for img_path in os.listdir("*.*"):
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify that the image is not corrupted
                records.append({"path": str(img_path), "label": label})
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return pd.DataFrame(records)    


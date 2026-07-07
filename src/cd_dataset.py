import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


LABEL2D = {"cats": 0, "dogs": 1}



class CatsDogsDataset(Dataset):
    """
    Custom Dataset class for loading cat and dog images.
    """

    def __init__(self, csv_path: str, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        label = LABEL2D[row["label"]]
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}
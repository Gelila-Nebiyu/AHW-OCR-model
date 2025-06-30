import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd

class OCRDataset(Dataset):
    def __init__(self, csv_file, img_dir, char2idx, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.char2idx = char2idx
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get filename and label
        img_name = self.annotations.iloc[idx, 0]
        label_text = self.annotations.iloc[idx, 1]

        # load image
        img_path = f"{self.img_dir}/{img_name}"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        # convert label to indexes
        label_indexes = [self.char2idx[c] for c in label_text if c in self.char2idx]

        label_tensor = torch.tensor(label_indexes, dtype=torch.long)

        return image, label_tensor

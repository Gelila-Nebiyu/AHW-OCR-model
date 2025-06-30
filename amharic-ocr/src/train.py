amharic_alphabet = list("ሀሁሂሃሄህሆአኡኢኣኤእኦ...")  # full set
char2idx = {c: idx + 1 for idx, c in enumerate(amharic_alphabet)}
import torchvision.transforms as T

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])
from src.dataset import OCRDataset

dataset = OCRDataset(
    csv_file="dataset/train/labels.csv",
    img_dir="dataset/train/images",
    char2idx=char2idx,
    transform=transform
)

print(f"Total samples: {len(dataset)}")
img, label = dataset[0]
print(img.shape)
print(label)

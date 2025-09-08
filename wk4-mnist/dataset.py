from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"] 
        label = sample["label"]

        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __repr__(self):
        return f"MNISTDataset(data={len(self.data)})"

import os
from torch.utils.data import Dataset
import PIL.Image as Image


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, n=None):
        self.root = root
        self.transform = transform
        self.img_list = os.listdir(root)[:n] if n is not None else os.listdir(root)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_list[index])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

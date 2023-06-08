import os
from torch.utils.data import Dataset
import PIL.Image as Image


class CustomDataset(Dataset):
    def __init__(self, root, labels=None, transform=None):
        self.root = root
        self.transform = transform
        self.img_list = os.listdir(root)
        if labels is not None:
            self.labels = labels

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_list[index])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        return image
    
class MySubDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample, label = self.data[index]
        # You can perform any necessary preprocessing on the sample and label here
        return sample, label

    def __len__(self):
        return len(self.data)

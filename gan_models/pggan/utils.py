import torch
import os
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image



class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_list = os.listdir(root)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_list[index])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image
    
class CustomDatasetWithLabels(Dataset):
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
    
    def __getlen__(self):
        return len(self.img_list)
    
class MySubDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample, label = self.data[index]
        # You can perform any necessary preprocessing on the sample and label here
        return sample, label

    def __len__(self):
        return len(self.data)
    
    def __getlen__(self):
        return len(self.data)
    
#class dataloader that returns the len of the dataset
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(MyDataLoader, self).__init__(dataset, batch_size, shuffle)
        
    def __len__(self):
        return self.dataset.__getlen__()



def gradient_penalty(critic, real, fake, alpha, step, split=None, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    
    interpolated_images = real * epsilon + fake.detach() * (1 - epsilon)
    interpolated_images.requires_grad_(True)
    # Calculate critic scores
    if split is None:
        mixed_scores = critic(interpolated_images, step, alpha)
    else:
        mixed_scores = critic(interpolated_images, step, alpha, split)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    
    return gp
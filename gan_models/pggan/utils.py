import torch
import os
from torch.utils.data import Dataset
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



def gradient_penalty(critic, real, fake, alpha, step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    
    interpolated_images = real * epsilon + fake.detach() * (1 - epsilon)
    interpolated_images.requires_grad_(True)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images, step, alpha)
    
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


    
import torch
import random
import numpy as np
import os
import torchvision
from torchvision.utils import save_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gradient_penalty(critic, real, fake, alpha, step, device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand([BATCH_SIZE, 1, 1, 1]).repeat(1, C, H, W).to(device)
    
    interpolated_images = real * epsilon + fake * (1 - epsilon)
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


    
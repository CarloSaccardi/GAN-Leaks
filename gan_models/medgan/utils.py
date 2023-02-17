import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def generator_loss(y_fake):
    """
    Gen loss
    Can be replaced with generator_loss = torch.nn.BCELoss(). Think why?
    """
    epsilon = 1e-12
    #loss = -0.5 * torch.mean(torch.log(y_fake + epsilon))
    loss = -torch.mean(torch.log(y_fake + epsilon))
    return loss


def autoencoder_loss(x_output, y_target, binary=True):
    """
    autoencoder_loss
    This implementation is equivalent to the following:
    torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then do the mean over the batch.
    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the later on, mean over both features and batches.
    """
    epsilon = 1e-12
    if binary:
        term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
        loss = torch.mean(-torch.sum(term, 1), 0)
    else:
        loss = torch.mean(torch.sum((x_output - y_target) ** 2, 1), 0)
    return loss



def discriminator_loss(outputs_real, outputs_fake):
    real_labels = torch.ones_like(outputs_real)
    fake_labels = torch.zeros_like(outputs_fake)
    real_loss = F.binary_cross_entropy_with_logits(outputs_real, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(outputs_fake, fake_labels)
    loss = real_loss + fake_loss
    return loss



def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples
    :param predicted: The predicted labels
    :param y_true: The gorund truth labels
    :return: Accuracy
    """
    predicted = predicted > 0.5
    accuracy = torch.mean((predicted == y_true).float())

    return accuracy


def sample_transform(sample):
    """
    Transform samples to their nearest integer
    :param sample: Rounded vector.
    :return:
    """
    sample[sample >= 0.5] = 1
    sample[sample < 0.5] = 0
    return sample


def init_weights(model, method='xavier_uniform', mean=0.0, std=0.02):
    if method == 'xavier_uniform':
        for module in model.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
    elif method == 'normal':
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=mean, std=std)
    else:
        raise ValueError("Invalid initialization method. Please choose 'xavier_uniform' or 'normal'.")





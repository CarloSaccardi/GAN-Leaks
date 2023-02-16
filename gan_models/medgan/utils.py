import torch
import torch.nn as nn
import torch.nn.init as init


def generator_loss(y_fake, y_true):
    """
    Gen loss
    Can be replaced with generator_loss = torch.nn.BCELoss(). Think why?
    """
    epsilon = 1e-12
    #loss = -0.5 * torch.mean(torch.log(y_fake + epsilon))
    return -torch.mean(torch.log(y_fake + epsilon))


def autoencoder_loss(x_output, y_target):
    """
    autoencoder_loss
    This implementation is equivalent to the following:
    torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then do the mean over the batch.
    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the later on, mean over both features and batches.
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss


def discriminator_loss(outputs_real, outputs_fake):
    """
    autoencoder_loss
    Cab be replaced with discriminator_loss = torch.nn.BCELoss(). Think why?
    """
    #loss = -torch.mean(labels * torch.log(outputs + 1e-12)) - torch.mean((1 - labels) * torch.log(1. - outputs + 1e-12))
    loss = -torch.mean(torch.log(outputs_real + 1e-12)) - torch.mean(torch.log(1. - outputs_fake + 1e-12))
    return loss


def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples
    :param predicted: The predicted labels
    :param y_true: The gorund truth labels
    :return: Accuracy
    """
    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    print('correct=', correct)
    accuracy = 100.0 * correct / total
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





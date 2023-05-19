import os
import numpy as np
import fnmatch
import PIL.Image
import matplotlib
import torchvision.transforms as transforms
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lpips_pytorch'))
import lpips_pytorch as ps

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NCOLS = 5


def check_folder(dir):
    '''
    create a new directory if it doesn't exist
    :param dir:
    :return:
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def save_files(save_dir, file_name_list, array_list):
    '''
    save a list of array with the given name
    :param save_dir: the directory for saving the files
    :param file_name_list: the list of the file names
    :param array_list: the list of arrays to be saved
    '''
    assert len(file_name_list) == len(array_list)

    for i in range(len(file_name_list)):
        np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)


def get_filepaths_from_dir(data_dir, ext):
    '''
    return all the file paths with extension 'ext' in the given directory 'data_dir'
    :param data_dir: the data directory
    :param ext: the extension type
    :return:
        path_list: list of file paths
    '''
    pattern = '*.' + ext
    path_list = []
    for d, s, fList in os.walk(data_dir):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                path_list.append(os.path.join(d, filename))
    return sorted(path_list)


def read_image(filepath, resolution=64, cx=89, cy=121):
    '''
    read,crop and scale an image given the path
    :param filepath:  the path of the image file
    :param resolution: desired size of the output image
    :param cx: x_coordinate of the crop center
    :param cy: y_coordinate of the crop center
    :return:
        image in range [-1,1] with shape (resolution,resolution,3)
    '''

    img = np.asarray(PIL.Image.open(filepath))
    shape = img.shape

    if shape == (resolution, resolution, 3):
        pass
    else:
        #reshape the image to the desired size
        img = PIL.Image.fromarray(img)
        img = img.resize((resolution, resolution))
        img = np.asarray(img)

    img = 2. * (img / 255.) - 1.

    return img


def read_image_norm(filepath, resolution=64, cx=89, cy=121):
    '''
    read,crop and scale an image given the path
    :param filepath:  the path of the image file
    :param resolution: desired size of the output image
    :param cx: x_coordinate of the crop center
    :param cy: y_coordinate of the crop center
    :return:
        image in range [-1,1] with shape (resolution,resolution,3)
    '''

    img = np.asarray(PIL.Image.open(filepath))
    shape = img.shape

    if shape == (resolution, resolution, 3):
        pass
    else:
        #reshape the image to the desired size
        img = PIL.Image.fromarray(img)
        img = img.resize((resolution, resolution))
        img = np.asarray(img)
    
    #normilize the image to be in range [-1,1]
    img = 2. * (img / 255.) - 1.
        
    return img


####################################################
## visualize
####################################################
def inverse_transform(imgs):
    '''
    normalize the image to be of range [0,1]
    :param imgs: input images
    :return:
        images with value range [0,1]
    '''
    imgs = (imgs + 1.) / 2.
    return imgs


def visualize_gt(imgs, save_dir):
    '''
    visualize the ground truth images and save
    :param imgs: input images with value range [-1,1]
    :param save_dir: directory for saving the results
    '''
    plt.figure(1)
    num_imgs = len(imgs)
    imgs = np.clip(inverse_transform(imgs), 0., 1.)
    NROWS = int(np.ceil(float(num_imgs) / float(NCOLS)))
    for i in range(num_imgs):
        plt.subplot(NROWS, NCOLS, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'input.png'))
    plt.close()


def visualize_progress(imgs, loss, save_dir, counter):
    '''
    visualize the optimization results and save
    :param imgs: input images with value range [-1,1]
    :param loss: the corresponding loss values
    :param save_dir: directory for saving the results
    :param counter: number of the function evaluation
    :return:
    '''
    plt.figure(2)
    num_imgs = len(imgs)
    imgs = np.clip(inverse_transform(imgs), 0., 1.)
    NROWS = int(np.ceil(float(num_imgs) / float(NCOLS)))
    for i in range(num_imgs):
        plt.subplot(NROWS, NCOLS, i + 1)
        plt.imshow(imgs[i])
        plt.title('loss: %.4f' % loss[i], fontdict={'fontsize': 8, 'color': 'blue'})
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'output_%d.png' % counter))
    plt.close()


def visualize_samples(img_r01, save_dir):
    plt.figure(figsize=(20, 20))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(img_r01[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples.png'))

### Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.0001
RANDOM_SEED = 1000




class Loss(torch.nn.Module):
    def __init__(self, distance, if_norm_reg=False):
        super(Loss, self).__init__()
        self.distance = distance
        self.lpips_model = ps.PerceptualLoss()
        self.if_norm_reg = if_norm_reg

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_lpips_fn = lambda x, y: 0.

        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(x, y, normalize=False).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])

    def forward(self, x_hat, x_gt):
        x_gt = torch.from_numpy(x_gt).float().reshape(1, 3, 64, 64).cuda()
        x_hat = torch.from_numpy(x_hat).float().reshape(1, 3, 64, 64).cuda()
        self.loss_lpips = self.loss_lpips_fn(x_hat, x_gt)
        self.loss_l2 = self.loss_l2_fn(x_hat, x_gt)
        self.vec_loss = LAMBDA2 * self.loss_lpips +  self.loss_l2
        return self.vec_loss

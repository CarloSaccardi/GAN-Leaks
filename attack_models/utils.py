import os
import numpy as np
import fnmatch
import PIL.Image
import matplotlib
import torchvision.transforms as transforms

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


def read_image(filepath, resolution=64):

  
    shape = np.asanyarray(PIL.Image.open(filepath)).shape

    if shape == (resolution, resolution, 3):
        img = np.array(PIL.Image.open(filepath))

    else:
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
        ])
        img = np.array(transform(PIL.Image.open(filepath)))


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

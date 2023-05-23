import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
from utils import *
from sklearn.neighbors import NearestNeighbors
import wandb
import yaml
import warnings

### Hyperparameters


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, default='debug',
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--syn_data_path', type=str,
                        help='directory to the synthetic data')
    parser.add_argument('--pos_data_dir', type=str, default=os.path.join(os.getcwd(), 'data', 'miniCelebA', 'train'),
                        help='the directory for the positive (training) query images set')
    parser.add_argument('--neg_data_dir', type=str, default=os.path.join(os.getcwd(), 'data', 'miniCelebA', 'test'),
                        help='the directory for the negative (testing) query images set')
    parser.add_argument('--data_num', '-dnum', type=int, default=20000,
                        help='the number of query images to be considered')
    parser.add_argument('--resolution', '-resolution', type=int, default=64,
                        help='generated image resolution')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--BATCH_SIZE', type=int, default=32)
    parser.add_argument('--local_config', type=str, default=None)
    parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")
    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## load dir
    assert os.path.exists(args.syn_data_path) 

    ## set up save_dir
    save_dir = os.path.join(os.getcwd(), 'fbb_attack', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir


#############################################################################################################
# main nearest neighbor search function
#############################################################################################################
def find_knn(nn_obj, query_imgs, args):
    '''
    :param nn_obj: Nearest Neighbor object
    :param query_imgs: query images
    :return:
        dist: distance between query samples to its KNNs among generated samples
        idx: index of the KNNs
    '''
    dist = []
    idx = []
    for i in tqdm(range(len(query_imgs) // args.BATCH_SIZE)):
        x_batch = query_imgs[i * args.BATCH_SIZE:(i + 1) * args.BATCH_SIZE]#(BATCH_SIZE, 64, 64, 3)
        x_batch = np.reshape(x_batch, [args.BATCH_SIZE, -1])#(BATCH_SIZE, 64*64*3)
        dist_batch, idx_batch = nn_obj.kneighbors(x_batch, args.K, return_distance=True)
        dist.append(dist_batch)#distance between query samples to its KNNs among generated samples
        idx.append(idx_batch)#index of the KNNs

    try:
        dist = np.concatenate(dist)
        idx = np.concatenate(idx)
    except:
        dist = np.array(dist)
        idx = np.array(idx)
    return dist, idx

def custom_knn(query_imgs, sample, loss, args):
    distances = []
    indices = []
    
    for i in range(len(query_imgs)):
        x_gt = torch.unsqueeze(query_imgs[i], 0)
        x_hat = torch.unsqueeze(sample, 0)
        distances.append(loss(x_hat, x_gt))
        indices.append(torch.tensor([i]))
    
    distances = torch.cat(distances)
    indices = torch.cat(indices)
    
    min_distance, min_index = torch.min(distances, dim=0)
    
    return min_distance.item(), indices[min_index].item()




def plot_closest_images(idx, query_imgs, syn_imgs, save_dir, class_type, num=20):
    '''
    plot the closest images
    :param idx: index of the KNNs
    :param query_imgs: query images
    :param syn_imgs: generated images
    :param save_dir: directory for saving the images
    :param num: number of closest images to be plotted
    :return:
    '''
    for i in range(num):
        query_img = syn_imgs[i]
        syn_img = query_imgs[idx[i][0]]
        img = np.concatenate((query_img, syn_img), axis=1)
        img = (img + 1.) / 2.
        PIL.Image.fromarray(np.uint8(img * 255)).save(os.path.join(save_dir, str(i) + class_type +'.png'))

#############################################################################################################
# main
#############################################################################################################
def main(args_):
    args, save_dir = check_args(args_)
    resolution = args.resolution

    ### load generated samples
    syn_data_paths = get_filepaths_from_dir(args.syn_data_path, ext='png')
    syn_imgs = np.array([read_image(f, resolution) for f in syn_data_paths])
    syn_imgs = torch.from_numpy(syn_imgs).float().permute(0, 3, 1, 2).to(device)


    ### load query images
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])
    pos_query_imgs = torch.from_numpy(pos_query_imgs).float().permute(0, 3, 1, 2).to(device)

    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])
    neg_query_imgs = torch.from_numpy(neg_query_imgs).float().permute(0, 3, 1, 2).to(device)

    ### load the distance function
    custom_loss = Loss('l2-lpips', if_norm_reg=False)

    pos_loss = []
    pos_idx = []

    neg_loss = []
    neg_idx = []

    for sample in tqdm(syn_imgs):

        pos_loss_item, pos_idx_item = custom_knn(pos_query_imgs, sample, custom_loss, args) #find closest positive image
        neg_loss_item, neg_idx_item = custom_knn(neg_query_imgs, sample, custom_loss, args) #find closest negative image
        print(pos_loss_item, neg_loss_item)

        pos_loss.append(pos_loss_item)
        pos_idx.append(pos_idx_item)
        neg_loss.append(neg_loss_item)
        neg_idx.append(neg_idx_item)

    pos_loss = np.array(pos_loss).reshape(-1, 1)
    pos_idx = np.array(pos_idx).reshape(-1, 1)

    neg_loss = np.array(neg_loss).reshape(-1, 1)
    neg_idx = np.array(neg_idx).reshape(-1, 1)

    ### positive query
    save_files(save_dir, ['pos_loss', 'pos_idx'], [pos_loss, pos_idx])
    plot_closest_images(pos_idx, pos_query_imgs, syn_imgs, save_dir, 'pos')

    ### negative query
    save_files(save_dir, ['neg_loss', 'neg_idx'], [neg_loss, neg_idx])
    plot_closest_images(neg_idx, neg_query_imgs, syn_imgs, save_dir, 'neg')


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  


if __name__ == '__main__':

    args = parse_arguments()
    if args.local_config is not None:
        with open(str(args.local_config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)
        if args.wandb:
            wandb_config = vars(args)
            run = wandb.init(project=str(args.wandb), entity="thesis_carlo", config=wandb_config)
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")

    main(args)

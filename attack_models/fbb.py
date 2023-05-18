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



#############################################################################################################
# main
#############################################################################################################
def main(args_):
    args, save_dir = check_args(args_)
    resolution = args.resolution

    ### load generated samples
    syn_data_paths = get_filepaths_from_dir(args.syn_data_path, ext='png')
    syn_imgs = np.array([read_image(f, resolution) for f in syn_data_paths])
    gen_feature = np.reshape(syn_imgs, [len(syn_imgs), -1])

    ### load query images
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])

    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])

    ### nearest neighbor search
    nn_obj = NearestNeighbors(n_neighbors=args.K)
    nn_obj.fit(gen_feature)#fitting NN classifier on the generated samples

    ### positive query
    pos_loss, pos_idx = find_knn(nn_obj, pos_query_imgs, args)
    save_files(save_dir, ['pos_loss', 'pos_idx'], [pos_loss, pos_idx])

    ### negative query
    neg_loss, neg_idx = find_knn(nn_obj, neg_query_imgs, args)
    save_files(save_dir, ['neg_loss', 'neg_idx'], [neg_loss, neg_idx])


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  


if __name__ == '__main__':

    args = parse_arguments()
    args.local_config = 'attack_models/config_attack.yaml'
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

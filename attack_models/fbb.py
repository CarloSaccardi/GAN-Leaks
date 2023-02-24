import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
from tools.utils import *
from sklearn.neighbors import NearestNeighbors

### Hyperparameters


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, default='debug',
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--syn_data_path', type=str, default='syn_data\dcgan\\npz_images\_2023_02_24__13_43_20\dcgan_synthetic_data.npz',
                        help='directory to the synthetic data')
    parser.add_argument('--noise_dir', type=str, default='syn_data\dcgan\\npz_noise\_2023_02_24__13_43_20\dcgan_noise.npz',
                        help='the directory for the noise')
    parser.add_argument('--pos_data_dir', type=str, default=os.path.join(os.getcwd(), 'data', 'miniCelebA', 'train'),
                        help='the directory for the positive (training) query images set')
    parser.add_argument('--neg_data_dir', type=str, default=os.path.join(os.getcwd(), 'data', 'miniCelebA', 'test'),
                        help='the directory for the negative (testing) query images set')
    parser.add_argument('--data_num', '-dnum', type=int, default=20000,
                        help='the number of query images to be considered')
    parser.add_argument('--resolution', '-resolution', type=int, default=64,
                        help='generated image resolution')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--BATCH_SIZE', type=int, default=10)
    return parser.parse_args()


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## load dir
    assert os.path.exists(args.syn_data_path) 
    assert os.path.exists(args.noise_dir)

    ## set up save_dir
    save_dir = os.path.join(os.getcwd(), 'fbb_attack', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir, args.syn_data_path, args.noise_dir


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
        x_batch = query_imgs[i * args.BATCH_SIZE:(i + 1) * args.BATCH_SIZE]
        x_batch = np.reshape(x_batch, [args.BATCH_SIZE, -1])
        dist_batch, idx_batch = nn_obj.kneighbors(x_batch, args.K)
        dist.append(dist_batch)
        idx.append(idx_batch)

    try:
        dist = np.concatenate(dist)
        idx = np.concatenate(idx)
    except:
        dist = np.array(dist)
        idx = np.array(idx)
    return dist, idx


def find_pred_z(gen_z, idx, args):
    '''
    :param gen_z: latent codes of the generated samples
    :param idx: index of the KNN
    :return:
        pred_z: predicted latent code
    '''
    pred_z = []
    for i in range(len(idx)):
        pred_z.append([gen_z[idx[i, nn]] for nn in range(args.K)])
    pred_z = np.array(pred_z)
    return pred_z


#############################################################################################################
# main
#############################################################################################################
def main():
    args, save_dir, load_dir_images, load_dir_noise = check_args(parse_arguments())
    resolution = args.resolution

    ### load generated samples
    generate = np.load(load_dir_images)
    noise = np.load(load_dir_noise)
    gen_imgs = generate['fake']
    gen_z = noise['noise']
    gen_feature = np.reshape(gen_imgs, [len(gen_imgs), -1])
    gen_feature = 2. * gen_feature - 1.

    ### load query images
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')[: args.data_num]
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])

    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')[: args.data_num]
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])

    ### nearest neighbor search
    nn_obj = NearestNeighbors(n_neighbors=args.K)
    nn_obj.fit(gen_feature)

    ### positive query
    pos_loss, pos_idx = find_knn(nn_obj, pos_query_imgs, args)
    pos_z = find_pred_z(gen_z, pos_idx, args)
    save_files(save_dir, ['pos_loss', 'pos_idx', 'pos_z'], [pos_loss, pos_idx, pos_z])

    ### negative query
    neg_loss, neg_idx = find_knn(nn_obj, neg_query_imgs, args)
    neg_z = find_pred_z(gen_z, neg_idx, args)
    save_files(save_dir, ['neg_loss', 'neg_idx', 'neg_z'], [neg_loss, neg_idx, neg_z])


if __name__ == '__main__':
    main()

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
import yaml
import warnings
import wandb


############################################################################
# visualization functions
############################################################################
def plot_roc(pos_results, neg_results):
    labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
    results = np.concatenate((neg_results, pos_results))
    fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
    auc = metrics.roc_auc_score(labels, results)
    ap = metrics.average_precision_score(labels, results)
    
    result_array = np.zeros_like(results)
    result_array[results > -0.14] = 1
    precision = metrics.precision_score(labels, result_array)
    
    return fpr, tpr, threshold, auc, ap, precision


def plot_hist(pos_dist, neg_dist, save_file):
    plt.figure()
    plt.hist(pos_dist, bins=100, alpha=0.5, weights=np.zeros_like(pos_dist) + 1. / pos_dist.size, label='positive')
    plt.hist(neg_dist, bins=100, alpha=0.5, weights=np.zeros_like(neg_dist) + 1. / neg_dist.size, label='negative')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel('distance')
    plt.ylabel('normalized frequency')
    plt.savefig(save_file)
    plt.close()


#############################################################################################################
# get the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_load_dir', '-ldir', type=str, default=None,
                        help='directory of the attack result')
    parser.add_argument('--attack_type', type=str, choices=['fbb', 'pbb', 'wb'],
                        help='type of the attack')
    parser.add_argument('--reference_load_dir', '-rdir', default=None,
                        help='directory for the reference model result (optional)')
    parser.add_argument('--save_dir', '-sdir', type=bool, default=True,
                        help='directory for saving the evaluation results (optional)')
    parser.add_argument("--wandb", default=None, help="Specify project name to log using WandB")
    parser.add_argument('--local_config', type=str, default=None)
    return parser.parse_args()


#############################################################################################################
# main
#############################################################################################################
def main():
    attack_type = args.attack_type
    result_load_dir = args.result_load_dir
    reference_load_dir = args.reference_load_dir
    save_dir = args.save_dir

    if attack_type == 'fbb':
        pos_loss = np.load(os.path.join(result_load_dir, 'pos_loss.npy'))#[:, 0]
        neg_loss = np.load(os.path.join(result_load_dir, 'neg_loss.npy'))#[:, 0]
    else:
        pos_loss = np.load(os.path.join(result_load_dir, 'pos_loss.npy')).flatten()
        neg_loss = np.load(os.path.join(result_load_dir, 'neg_loss.npy')).flatten()

    #idx = np.nonzero(pos_loss < 0.2)[0]
    #pos_loss = pos_loss[idx]
    #neg_loss = neg_loss[idx]
    ### plot roc curve
    fpr, tpr, threshold, auc, ap, precision = plot_roc(-pos_loss, -neg_loss)
    plt.plot(fpr, tpr, label='%s attack, auc=%.3f, ap=%.3f' % (attack_type, auc, ap))
    print("The AUC ROC value of %s attack is: %.3f " % (attack_type, auc))
    print("The precision of %s attack is: %.3f " % (attack_type, precision))

    ################################################################
    # attack calibration
    ################################################################
    if reference_load_dir is not None:
        pos_ref = np.load(os.path.join(reference_load_dir, 'pos_loss.npy'))
        neg_ref = np.load(os.path.join(reference_load_dir, 'neg_loss.npy'))

        num_pos_samples = np.minimum(len(pos_loss), len(pos_ref))
        num_neg_samples = np.minimum(len(neg_loss), len(neg_ref))

        try:
            pos_calibrate = pos_loss[:num_pos_samples] - pos_ref[:num_pos_samples]
            neg_calibrate = neg_loss[:num_neg_samples] - neg_ref[:num_neg_samples]

        except:
            pos_calibrate = pos_loss[:num_pos_samples] - pos_ref[:num_pos_samples, 0]
            neg_calibrate = neg_loss[:num_neg_samples] - neg_ref[:num_neg_samples, 0]

        fpr, tpr, threshold, auc, ap = plot_roc(-pos_calibrate, -neg_calibrate)
        plt.plot(fpr, tpr, label='calibrated %s attack, auc=%.3f, ap=%.3f' % (attack_type, auc, ap))
        print("The AUC ROC value of calibrated %s attack is: %.3f " % (attack_type, auc))

    plt.legend(loc='lower right')
    plt.xlabel('false positive')
    plt.ylabel('true positive')
    plt.title('ROC curve')

    #push to wandb
    if args.wandb:
        wandb.log({"roc": wandb.Image(plt)})
        wandb.log({"AUC ROC value": auc})
        wandb.log({"AP": ap})
        wandb.log({"fpr": fpr})
        wandb.log({"tpr": tpr})
        wandb.log({"threshold": threshold})

    if save_dir:
        plt.savefig(os.path.join(result_load_dir, 'roc.png'))
    plt.show()


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  


if __name__ == '__main__':

    args = parse_arguments()
    #args.local_config = 'attack_models/attack_eval.yaml'
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

    main()

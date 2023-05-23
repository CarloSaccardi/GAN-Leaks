import os
import shutil
import argparse
import yaml
import warnings
import PIL.Image
import random
import numpy as np
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_images', type=float, default=2048,
                    help='the number of images to move to the two directories')
parser.add_argument('--identity_annotations', type=str, default='data/identities_ann.txt',
                    help='the path to the identity annotations file')
parser.add_argument('--input_dir', type=str, default='data/img_align_celeba',
                    help='the path to the input directory')
parser.add_argument('--output_dir1', type=str, default='data/celebAhuge_positive',
                    help='the path to the first output directory')
parser.add_argument('--output_dir2', type=str, default='data/celebAhuge_negative',
                    help='the path to the second output directory')
parser.add_argument('--output_dir3', type=str, default='data/celebAhuge_positive_64',
                    help='the path to the first output directory')
parser.add_argument('--img_size', type=int, default=64, 
                    help='the height / width of the input image to network')
parser.add_argument('--local_config', default=None, help='path to config file')

args = parser.parse_args()

def main():

    #Read the identity annotations file
    diz = {}
    with open(args.identity_annotations) as f:
        for line in f:
            annotation, identity = line.strip().split()
            diz.setdefault(annotation, []).append(identity)


    identities = list(diz.keys())
    private_identities = identities[:len(identities)//2]
    public_identities = identities[len(identities)//2:]
    private_images = []
    public_images = []
    for identity in private_identities:
        if len(private_images) < args.num_images:
            if ( args.num_images-len(private_images) ) > len(diz[identity]):
                private_images += diz[identity]
            else:
                private_images += diz[identity][:args.num_images-len(private_images)]
        else:
            break

    for identity in public_identities:
        if len(public_images) < args.num_images:
            if ( args.num_images-len(public_images) ) > len(diz[identity]):
                public_images += diz[identity]
            else:
                public_images += diz[identity][:args.num_images-len(public_images)]
        else:
            break

    assert any(img in private_images for img in public_images) == False, 'The two lists are not disjoint!'
    assert any(ann in private_identities for ann in public_identities) == False, 'The two lists are not disjoint!'

    # Create output directories
    if os.path.exists(args.output_dir1):
        shutil.rmtree(args.output_dir1)

    if os.path.exists(args.output_dir2):
        shutil.rmtree(args.output_dir2) 

    if os.path.exists(args.output_dir3):
        shutil.rmtree(args.output_dir3)   

    os.makedirs(args.output_dir1, exist_ok=True)
    os.makedirs(args.output_dir2, exist_ok=True)
    os.makedirs(args.output_dir3, exist_ok=True)

    # Move the first part of images to output_dir1
    for img in private_images:
        img_id = img.split('.')[0]
        src_path = os.path.join(args.input_dir, img)
        dst_path = os.path.join(args.output_dir1, img_id + '.png')
        dst_path_64 = os.path.join(args.output_dir3, img_id + '.png')
        center_crop(src_path, dst_path)
        resize_image(dst_path, dst_path_64)

    # Move the second part of images to output_dir2
    for img in public_images:
        img_id = img.split('.')[0]
        src_path = os.path.join(args.input_dir, img)
        dst_path = os.path.join(args.output_dir2, img_id + '.png')
        center_crop(src_path, dst_path)




def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  

def resize_image(src_path, dst_path):
    img = PIL.Image.open(src_path)
    img = img.resize((args.img_size, args.img_size))
    img.save(dst_path)


def center_crop(src_path, dst_path, cx=89, cy=121):
    img = np.asarray(PIL.Image.open(src_path))
    assert img.shape == (218, 178, 3)
    img = img[cy - 64: cy + 64, cx - 64: cx + 64]
    PIL.Image.fromarray(img).save(dst_path)
    


if __name__ == '__main__':

    if args.local_config is not None:

        with open(str(args.local_config), "r") as f:
            config = yaml.safe_load(f)
        update_args(args, config)

        if not args.ailab:
            import wandb #wandb is not supported on ailab server 
            
            # update_args(args, dict(run.config))
    else:
        warnings.warn("No config file was provided. Using default parameters.")

    main()
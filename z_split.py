import os
import shutil
import argparse
import yaml
import warnings
import PIL.Image
import torch
import torchvision
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_images', type=float, default=20000,
                    help='the number of images to move to the two directories')
parser.add_argument('--input_dir', type=str, default='data/celebA_aligned',
                    help='the path to the input directory')
parser.add_argument('--output_dir1', type=str, default='data/celebAhuge_positive',
                    help='the path to the first output directory')
parser.add_argument('--output_dir2', type=str, default='data/celebAhuge_negative',
                    help='the path to the second output directory')
parser.add_argument('--img_size', type=int, default=64, 
                    help='the height / width of the input image to network')
parser.add_argument('--local_config', default=None, help='path to config file')

args = parser.parse_args()

def main():
    #download celeba with torchviosn 
    
    # Get list of image file names in the input directory
    img_list = os.listdir(args.input_dir)

    # Calculate the split index


    # Split the list into two parts
    img_list1 = img_list[:args.num_images]
    img_list2 = img_list[args.num_images:args.num_images * 2]

    assert any(img in img_list1 for img in img_list2) == False, 'The two lists are not disjoint!'

    # Create output directories
    #if os.path.exists(args.output_dir1) then delete every file in it

    if os.path.exists(args.output_dir1):
        shutil.rmtree(args.output_dir1)

    if os.path.exists(args.output_dir2):
        shutil.rmtree(args.output_dir2)    

    os.makedirs(args.output_dir1, exist_ok=True)
    os.makedirs(args.output_dir2, exist_ok=True)

    # Move the first part of images to output_dir1
    for img in img_list1:
        src_path = os.path.join(args.input_dir, img)
        dst_path = os.path.join(args.output_dir1, img)
        resize_image(src_path, dst_path)

    # Move the second part of images to output_dir2
    for img in img_list2:
        src_path = os.path.join(args.input_dir, img)
        dst_path = os.path.join(args.output_dir2, img)
        resize_image(src_path, dst_path)



def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  

def resize_image(src_path, dst_path):
    img = PIL.Image.open(src_path)
    img = img.resize((args.img_size, args.img_size))
    img.save(dst_path)

#align images to each other based on facial landmarks
def align_images():
    pass
    
  


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
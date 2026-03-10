## Project: Age-related macular degeneration (AMD) binary classification of retinal fundus images
## Written by: Dr. Adrian Agaldran
## Re-written by: Dr. Waziha Kabir
## Date of last modification: Dec 13, 2021 
## Code summary: 
              ## Input: Fundus images from a directory
	      ## Output: Pre-processed fundus images saved in a directory


import os, sys, argparse
import os.path as osp
import numpy as np
from utils.get_mask import crop_to_fov
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm
from skimage.measure import regionprops


parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='data/images', help='path to training data')
parser.add_argument('--im_path_out', type=str, default='data/cropped_images/', help='path data')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')



if __name__ == '__main__':

    args = parser.parse_args()
    im_path_out = args.im_path_out
    im_path = args.im_path
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')


    os.makedirs(im_path_out, exist_ok=True)
    im_list = sorted(os.listdir(im_path))
    rsz = Resize(tg_size)
    print('total amount of images = {}'.format(len(im_list)))
    print('to be stored at {}'.format(im_path_out))
    for i in tqdm(range(len(im_list))):
        im_name = im_list[i]
        img = Image.open(osp.join(im_path, im_name))
        img_proc = crop_to_fov(img)
        img_proc = rsz(img_proc)
        img_proc.save(osp.join(im_path_out,im_name))

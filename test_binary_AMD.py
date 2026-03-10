## Project: Age-related macular degeneration (AMD) binary classification of retinal fundus images
## Written by: Dr. Adrian Agaldran
## Re-written by: Dr. Waziha Kabir
## Date of last modification: Dec 13, 2021 
## Code summary: 
              ## Input: Test pre-processed fundus images from a directory & the AMD model
              ## Output: Prediction of AMD or No-AMD with testing time/image


import argparse, time
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_cls_loader

from utils.evaluation import evaluate_bin_cls
from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model
from tqdm import trange
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import os.path as osp
import os
import sys


def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/cropped_images/', help='path data')
parser.add_argument('--csv_test', type=str, default='data/test_amd.csv', help='path to test data csv')
parser.add_argument('--model_name', type=str, default='bit_resnext101_1', help='selected architecture')
parser.add_argument('--load_path', type=str, default='AMDmodels/v1.0.0.0/cycle_14_mAUC_95.33_MCC_89.73_F1_86.23', help='path to saved model')
parser.add_argument('--dihedral_tta', type=int, default=1, help='dihedral group cardinality (0)')
parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='512,512')
parser.add_argument('--n_classes', type=int, default=1, help='number of target classes (1)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--results_path', type=str, default='results/', help='path to output csv')
parser.add_argument('--csv_out_test', type=str, default='results_test.csv', help='path to output csv')

args = parser.parse_args()


def run_one_epoch_cls(loader, model, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, inputs in enumerate(loader):
            if loader.dataset.has_labels:
                (inputs, labels, _) = inputs
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            else:
                inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs = logits.sigmoid()

            probs = probs.detach().cpu().numpy().squeeze()
            if probs.ndim == 0: # handles case of 1-item batch
                probs = np.expand_dims(probs, axis=0)
            preds = probs > 0.1292153 ## Waziha-Riadh threshold
	       
            probs_all.extend(probs)
            preds_all.extend(preds)
            
            if loader.dataset.has_labels:
                labels_all.extend(labels.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    if loader.dataset.has_labels:
        return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all)
    return np.stack(preds_all), np.stack(probs_all), None

def test_cls_tta_dihedral(model, test_loader, n=3):
    probs_tta = []
    prs = [0, 1]

    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs)
                
    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = probs_tta > 0.1292153 ## Waziha-Riadh threshold
    
    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)
        
    del model
    torch.cuda.empty_cache()
    return test_probs, test_preds, test_labels


if __name__ == '__main__':
    '''
    Example:
    python test_dihedral_tta.py --csv_val data/val.csv --csv_test data/test.csv --n_classes 6 --dihedral_tta 1 --batch_size 8
    --load_path experiments/bit_512_shorter/cycle_03_MCC_74.31_F1_86.51_AUC_93.61 --csv_out_val borrar_val.csv --csv_out_test borrar_test.csv
    '''
    data_path = args.data_path
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name
    load_path = args.load_path
    results_path = osp.join(args.results_path, load_path.split('/')[1], load_path.split('/')[2])
    os.makedirs(results_path, exist_ok=True)
    bs = args.batch_size
    csv_test = args.csv_test
    n_classes = args.n_classes
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')
    dihedral_tta = args.dihedral_tta
    csv_out_test = args.csv_out_test

    print('* Loading model {} from {}'.format(model_name, load_path))
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    model, stats = load_model(model, load_path, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    start = time.time() #Waziha start
    print('* Creating Test Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_cls_loader(data_path=data_path, csv_path_test=csv_test,  batch_size=bs, mean=mean, std=std, tg_size=tg_size, test=True)
    end = time.time() #Waziha end

    if dihedral_tta==0:
        probs, preds, labels = test_cls(model, test_loader)
    elif dihedral_tta>0:
        probs, preds, labels = test_cls_tta_dihedral(model, test_loader, n=dihedral_tta)
    else: sys.exit('dihedral_tta must be >=0')

    if n_classes == 1:
        im_list = list(test_loader.dataset.im_list)
        im_list = [n.replace('data/cropped_images/','') for n in im_list]
        df_nogt = pd.DataFrame(zip(im_list, probs, preds), columns=['image_id', 'AMD', 'Prediction'])
    else: sys.exit('Wrong number of classes when saving dataframe')
    df_nogt.to_csv(osp.join(results_path, csv_out_test), index=False)
#Waziha start
    for i in range(len(preds)):
       print("DATA_TAG:=AMD={};".format(preds[i]),"File={}".format(im_list[i]))

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('\nTesting time per image: {:0>2}h {:0>2}min {:05.6f}secs'.format(int(hours/len(preds)), int(minutes/len(preds)), seconds/len(preds)))
#Waziha end

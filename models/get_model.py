import sys, os
import numpy as np
import torch
from . import bit_models, bit_models_MOD
import timm
from torchvision.models import resnet, mobilenet_v2
from .repvgg import repvgg_model_convert, create_RepVGG_A0, create_RepVGG_A1, create_RepVGG_B1g4


def cum_derivative_left(tens_1d):
    p_cum = torch.nn.functional.pad(tens_1d, pad=(1,0))
    return (p_cum-p_cum.roll(shifts=1))[:, 1:]

# IMAGENET PERFORMANCE FOR TIMM MODELS IS HERE:
# https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    elif model_name == 'bit_resnext101_1':
        bit_variant = 'BiT-M-R101x1'
        model = bit_models.KNOWN_MODELS[bit_variant](head_size=n_classes, zero_head=True)
        if not os.path.isfile('models/BiT-M-R101x1.npz'):
            print('downloading bit_resnext101_1 weights:')
            os.system('wget https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz -P models/')
        model.load_from(np.load('models/BiT-M-R101x1.npz'))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    else:
        sys.exit('not a valid model_name, check models.get_model.py')
    setattr(model, 'n_classes', n_classes)
    setattr(model, 'cum_derivative_left', cum_derivative_left)

    return model, mean, std



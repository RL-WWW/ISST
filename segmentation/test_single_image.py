###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import sys

import torch
import torchvision.transforms as transform

import encoding.utils as utils

from PIL import Image

from encoding.nn import BatchNorm
from encoding.datasets import datasets
from encoding.models import get_segmentation_model, MultiEvalModule

from .option import Options
import numpy as np
from L0_serial import L0_smooth


def semseg(input_path, output_path=None, with_L0=False):
    """
    param:
        input_path: str, path of input image
        output_path: str, path to save output image
    return: tuple, [animal_name, "background"] if pixels of "background" dominate,
                   ["background", animal_name] else.
    """
    sys.argv = sys.argv[:1]
    option = Options()
    args = option.parse()
    args.aux = True
    args.se_loss = True
    args.resume = "./checkpoints/encnet_jpu_res101_pcontext.pth.tar"  # model checkpoint
    torch.manual_seed(args.seed)

    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    # using L0_smooth to transform the orignal picture
    if with_L0:
        mid_result = os.path.join(os.path.dirname(input_path), "L0_result.png")
        L0_smooth(input_path, mid_result)
        input_path = mid_result

    # model
    model = get_segmentation_model(args.model,
                                   dataset=args.dataset,
                                   backbone=args.backbone,
                                   dilated=args.dilated,
                                   lateral=args.lateral,
                                   jpu=args.jpu,
                                   aux=args.aux,
                                   se_loss=args.se_loss,
                                   norm_layer=BatchNorm,
                                   base_size=args.base_size,
                                   crop_size=args.crop_size)
    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(
            args.resume))
    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("semseg model loaded successfully!")
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    num_classes = datasets[args.dataset.lower()].NUM_CLASS
    evaluator = MultiEvalModule(model,
                                num_classes,
                                scales=scales,
                                flip=args.ms).cuda()
    evaluator.eval()
    classes = np.array([
        'empty', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
        'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet', 'car',
        'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow', 'cup',
        'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass',
        'ground', 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
        'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
        'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
        'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
        'window', 'wood'
    ])
    animals = ['bird', 'cat', 'cow', 'dog', 'horse', 'mouse', 'sheep']
    img = input_transform(
        Image.open(input_path).convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        output = evaluator.parallel_forward(img)[0]
        predict = torch.max(output, 1)[1].cpu().numpy() + 1
    pred_idx = np.unique(predict)
    pred_label = classes[pred_idx]
    print("[SemSeg] ", input_path, ": ", pred_label, sep='')

    main_pixels = 0
    main_idx = -1
    for idx, label in zip(pred_idx, pred_label):
        if label in animals:
            pixels = np.sum(predict == idx)
            if pixels > main_pixels:
                main_pixels = pixels
                main_idx = idx
    background_pixels = np.sum(predict != main_idx)

    main_animal = classes[main_idx]
    predict[predict != main_idx] = 29
    mask_matrix = predict.copy()

    if output_path is not None:
        mask_matrix[np.where(mask_matrix != 29)] = 1
        mask_matrix[np.where(mask_matrix == 29)] = 0
        mask = utils.get_mask_pallete(mask_matrix, args.dataset)
        mask.save(output_path)

    if main_idx < 29:
        return predict, (main_animal, "background")
    else:
        return predict, ("background", main_animal)


if __name__ == "__main__":
    input_path = './images/test/input/horse_man.jpg'
    output_path = './images/test/output/horse_man.png'
    print(semseg(input_path, output_path))

from PIL import Image
import torch
import numpy as np
import os
import torchvision
from data import create_dataset
from experiments.segmentation.test_single_image import semseg
from models import create_model
from util import util
from util.visualizer import save_images
import torchvision.transforms as tfs
from options.test_options import TestOptions
from classes_to_models import get_mapping
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image", default="C:/Users/ls/Desktop/exp/horse_3.png", help="The image we need to transfer")
    parser.add_argument("--image", default="C:/Users/ls/Desktop/exp/horse_2.jpeg", help="The original image")
    parser.add_argument("--folder", default="C:/Users/ls/Desktop/exp/input", help="input folder")
    parser.add_argument("--temp_folder", default=None, help='Where to put the middle images')
    parser.add_argument("--target_path", default=None, help='Where to put the final result')
    return parser.parse_args()

def config_models_and_datasets(model_names):
    opt = None
    models = dict()
    for name in model_names:
        sys.argv = sys.argv[:1]
        sys.argv.append("--dataroot")
        sys.argv.append(folder)
        sys.argv.append("--name")
        sys.argv.append(name)
        sys.argv.append("--model")
        sys.argv.append("test")
        sys.argv.append("--no_dropout")
        opt = TestOptions().parse()
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_sized = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

        model = create_model(opt)
        model.setup(opt)
        models[name] = model
        print(f"model {name} instantiated")
    dataset = create_dataset(opt)
    return dataset, models


if __name__ == '__main__':
    args = parse_args()
    totensor = tfs.ToTensor()
    toimage = tfs.ToPILImage()

    # config folder to put the output
    folder = args.temp_folder if args.temp_folder is not None else os.path.dirname(args.image)
    folder = os.path.join(folder, "temp")

    # 读入图片进行分离
    original_im = Image.open(args.image)
    original_img = totensor(original_im)

    matrix, image_classes = semseg(args.image, os.path.join(folder, "mask.png"))
    # matrix and classes of the processed images
    img = torch.tensor(matrix, dtype=torch.float32)
    img = totensor(toimage(img).convert("RGB"))
    img_2_dim = torch.sum(img, dim=0)

    if not os.path.exists(folder):
        os.mkdir(folder)

    classes, models, classes_to_models = get_mapping()
    # map class to a list of models

    target_path = args.target_path if args.target_path is not None else os.path.join(os.path.dirname(args.image), "results")

    toimage(original_img).save(os.path.join(folder, f"exp_real.png"))
    for i, pixel in enumerate(torch.unique(img_2_dim).sort()[0]):
        # plane = torch.zeros_like(img)
        plane = original_img
        # if image_classes[i] != 'background':
        #     plane = original_img
        # else:
        #     for j in range(len(plane)):
        #         plane[j][torch.where(img_2_dim == pixel)] = original_img[j][torch.where(img_2_dim == pixel)]
        toimage(plane).save(os.path.join(folder, f"exp{i}.png"))

    # 调用cycleGAN
    model_names = []
    for class_name in image_classes:
        for model_name in classes_to_models[class_name]:
            if model_name not in model_names:
                model_names.append(model_name)
    print("total model names:")
    print(model_names)
    dataset, models = config_models_and_datasets(model_names)

    # 生成Baseline结果
    for name in model_names:
        model = models[name]
        for data in dataset:
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()

            fake_im = toimage(util.tensor2im(visuals['fake']))
            fake_im = torchvision.transforms.functional.resize(fake_im, (img.shape[1], img.shape[2]), interpolation=2)
            fake_im.save(os.path.join(target_path, f"baseline_{name}.png"))
            break

    shape = []
    num = 1
    for class_name in image_classes:
        if class_name == 'background':
            shape.append(len(classes_to_models[class_name]) + 1)
        else:
            shape.append(len(classes_to_models[class_name]))
        num *= shape[-1]

    indexes = torch.arange(num).reshape(shape)
    new_images = torch.zeros(num, *(img.shape))

    for i, (pixel, data, class_name) in enumerate(zip(torch.unique(img_2_dim).sort()[0], dataset, image_classes)):
        # get models
        buffer = 0
        if class_name == 'background':
            buffer = 1
            image_index = indexes.transpose(0, i)[0].flatten()

            real_im = toimage(util.tensor2im(data['A']))
            real = totensor(
                torchvision.transforms.functional.resize(real_im, (img.shape[1], img.shape[2]), interpolation=2))

            for k in range(3):
                for idx in image_index:
                    new_images[idx][k][torch.where(img_2_dim == pixel)] =real[k][torch.where(img_2_dim == pixel)]


        for j, model_name in enumerate(classes_to_models[class_name]):
            model = models[model_name]

            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()

            # A_path = data['A_paths'][0].split('.')
            # A_path[0] = A_path[0] + '_fake'
            # A_path = '.'.join(A_path)
            # toimage(visuals['fake'].squeeze()).save(A_path)
            # for i,pixel in enumerate(torch.unique(img_2_dim)):
            #     fake_im = Image.open(f"exp/exp{i}_fake.png")
            fake_im = toimage(util.tensor2im(visuals['fake']))
            fake_im = torchvision.transforms.functional.resize(fake_im, (img.shape[1], img.shape[2]), interpolation=2)
            # fake_im.save(os.path.join(target_path, f"fake_{i}.png"))
            fake = totensor(fake_im)

            image_index = indexes.transpose(0, i)[j + buffer].flatten()

            for k in range(3):
                for idx in image_index:
                    new_images[idx][k][torch.where(img_2_dim == pixel)] = fake[k][torch.where(img_2_dim == pixel)]

    # new_image[torch.where(real_img == pixel)] = fake[torch.where(real_img == pixel)]
    # print(torch.where(real_img == pixel))
    for i, new_image in enumerate(new_images):
        result = toimage(new_image)
        result.save(os.path.join(target_path, f"result_{i}.png"))

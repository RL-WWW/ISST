from PIL import Image
import torch
import numpy as np
import os
import torchvision
from data import create_dataset
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
    parser.add_argument("--image", default="C:/Users/ls/Desktop/exp/horse_3.png", help="The image we need to transfer")
    parser.add_argument("--real_image", default="C:/Users/ls/Desktop/exp/horse_2.jpg", help="The original image")
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

    im = Image.open(args.image).convert("RGB")
    original_im = Image.open(args.real_image)

    # 读入图片进行分离
    img = totensor(im)
    original_img = totensor(original_im)
    img_2_dim = torch.sum(img, axis=0)

    # config folder to put the output
    folder = args.temp_folder if args.temp_folder is not None else os.path.dirname(args.image)
    folder = os.path.join(folder, "temp")
    if not os.path.exists(folder):
        os.mkdir(folder)

    classes, models, classes_to_models = get_mapping()
    # map class to a list of models

    # image_classes = ["bag", "bed", "car"]
    image_classes = ['sky', "horse"]  # classes of the processed images

    target_path = args.target_path if args.target_path is not None else os.path.join(os.path.dirname(args.image), "results")

    for i, pixel in enumerate(torch.unique(img_2_dim)):
        plane = torch.zeros_like(img)
        for j in range(len(plane)):
            plane[j][torch.where(img_2_dim == pixel)] = original_img[j][torch.where(img_2_dim == pixel)]
        toimage(plane).save(os.path.join(folder, f"exp{i}.png"))

    # 调用cycleGAN
    model_names = []
    for _, model_for_class in classes_to_models.items():
        for model_name in model_for_class:
            if model_name not in model_names:
                model_names.append(model_name)
    print("total model names:")
    print(model_names)
    dataset, models = config_models_and_datasets(model_names)

    shape = []
    num = 1
    for class_name in image_classes:
        shape.append(len(classes_to_models[class_name]))
        num *= shape[-1]
    indexes = torch.arange(num).reshape(shape)
    new_images = torch.zeros(num, *(img.shape))

    for i, (pixel, data, class_name) in enumerate(zip(torch.unique(img_2_dim), dataset, image_classes)):
        # get models
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
            fake = totensor(torchvision.transforms.functional.resize(fake_im, (img.shape[1], img.shape[2]), interpolation=2))

            image_index = indexes.transpose(0, i)[j].flatten()

            for k in range(3):
                for idx in image_index:
                    new_images[idx][k][torch.where(img_2_dim == pixel)] = fake[k][torch.where(img_2_dim == pixel)]

    #     new_image[torch.where(real_img == pixel)] = fake[torch.where(real_img == pixel)]
    #     print(torch.where(real_img == pixel))
    for i, new_image in enumerate(new_images):
        result = toimage(new_image)
        result.save(os.path.join(target_path, f"result_{i}.png"))

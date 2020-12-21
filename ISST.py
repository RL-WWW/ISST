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
    parser.add_argument("--image", default="C:/Users/ls/Desktop/exp/horse0.jpeg", help="The image we need to transfer")
    # parser.add_argument("--image", default=None, help="The original image")
    parser.add_argument("--folder", default="C:\\Users\\ls\\Desktop\\animal_pics\\horse", help="input folder")
    parser.add_argument("--temp_folder", default=None, help='Where to put the middle images')
    parser.add_argument("--target_path", default=None, help='Where to put the final result')
    return parser.parse_args()

def config_models(model_names, folder):
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
    return models

def config_models_and_datasets(model_names, folder):
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

def config_datasets():
    sys.argv = sys.argv[:1]
    sys.argv.append("--dataroot")
    sys.argv.append(temp_folder)
    sys.argv.append("--model")
    sys.argv.append("test")
    sys.argv.append("--no_dropout")
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_sized = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    return create_dataset(opt)


def ISST(input_path, folder, folder2, target_path, models):
    # 读入图片进行分离
    original_im = Image.open(input_path)
    original_img = totensor(original_im)

    matrix, image_classes = semseg(input_path, os.path.join(folder2, "mask.png"), with_L0=False)
    # matrix and classes of the processed images
    img = torch.tensor(matrix, dtype=torch.float32)
    img = totensor(toimage(img).convert("RGB"))
    img_2_dim = torch.sum(img, dim=0)

    if not os.path.exists(folder):
        os.mkdir(folder)

    classes, _, classes_to_models = get_mapping()
    # map class to a list of models

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
    # dataset, models = config_models_and_datasets(model_names, temp_folder)
    dataset = config_datasets()
    # 生成Baseline结果
    for name in model_names:
        model = models[name]
        for data in dataset:
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            # img_path = model.get_image_paths()

            fake_im = toimage(util.tensor2im(visuals['fake']))
            fake_im = torchvision.transforms.functional.resize(fake_im, (img.shape[1], img.shape[2]), interpolation=2)
            fake_im.save(target_path+f"{name}.png")
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

            toimage(real).save(os.path.join(folder2, 'background_0.png'))
            for k in range(3):
                for idx in image_index:
                    new_images[idx][k][torch.where(img_2_dim == pixel)] = real[k][torch.where(img_2_dim == pixel)]

        for j, model_name in enumerate(classes_to_models[class_name]):
            model = models[model_name]

            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            # img_path = model.get_image_paths()

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

            if class_name == 'background':
                toimage(fake).save(os.path.join(folder2, 'background_1.png'))
            else:
                toimage(fake).save(os.path.join(folder2, 'animal.png'))

    # new_image[torch.where(real_img == pixel)] = fake[torch.where(real_img == pixel)]
    # print(torch.where(real_img == pixel))
    for i, new_image in enumerate(new_images):
        result = toimage(new_image)
        result.save(target_path+f'{i}.png')



if __name__ == '__main__':
    args = parse_args()
    totensor = tfs.ToTensor()
    toimage = tfs.ToPILImage()

    # config folder to put the output
    if args.image is None:
        temp_folder = args.temp_folder if args.temp_folder is not None else os.path.dirname(args.folder)
        target_folder = args.target_path if args.target_path is not None else os.path.join(os.path.dirname(args.folder),
                                                                                    "results")
    else:
        temp_folder = args.temp_folder if args.temp_folder is not None else os.path.dirname(args.image)
        target_folder = args.target_path if args.target_path is not None else os.path.join(os.path.dirname(args.image),
                                                                                    "results")
    temp_folder = os.path.join(temp_folder, "temp")
    temp_folder2 = os.path.join(temp_folder, "temp2")
    if not os.path.exists(temp_folder): os.mkdir(temp_folder)
    if not os.path.exists(temp_folder2): os.mkdir(temp_folder2)
    if not os.path.exists(target_folder): os.mkdir(target_folder)

    # model_names = ['horse2zebra_pretrained', 'winter2summer_yosemite_pretrained', 'style_cezanne_pretrained', 'style_monet_pretrained', 'style_ukiyoe_pretrained', 'style_vangogh_pretrained']
    model_names = ['horse2zebra_pretrained', 'winter2summer_yosemite_pretrained']
    models = config_models(model_names, temp_folder)

    if args.image is not None:
        ISST(args.image, temp_folder, temp_folder2, os.path.join(target_folder, os.path.basename(args.image).split(".")[0] + '_fake'), models)
    else:
        assert args.folder is not None
        for i, input_path in enumerate(os.listdir(args.folder)):
            ISST(os.path.join(args.folder, input_path), temp_folder, temp_folder2, os.path.join(target_folder, os.path.basename(input_path).split(".")[0]  + '_fake'), models)
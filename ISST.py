from PIL import Image
import torch
import numpy as np
import os
import torchvision
from data import create_dataset
from segmentation.test_single_image import semseg
from models import create_model
from util import util
from util.visualizer import save_images
import torchvision.transforms as tfs
from options.test_options import TestOptions
from classes_to_models import get_mapping
import argparse
import mix
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image", default="C:/Users/ls/Desktop/exp/horse1.jpeg", help="The image we need to transfer")
    parser.add_argument("--image", default="C:\\Users\\ls\\Desktop\\animal_pics\\dog\\dog1.jpeg", help="The original image")
    parser.add_argument("--folder", default=None, help="input folder")
    parser.add_argument("--temp_folder", default=None, help='Where to put the middle images')
    parser.add_argument("--target_path", default="C:\\Users\\ls\\Desktop\\animal_pics\\dog_results", help='Where to put the final result')
    # parser.add_argument("--target_path", default=None, help='Where to put the final result')
    parser.add_argument("--with_L0", default=False, action='store_true',help='whether to use L0-smooth before segmentation')
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

    matrix, image_classes = semseg(input_path, os.path.join(folder2, "mask.png"), with_L0=args.with_L0)
    matrix = torch.tensor(matrix[0])
    unique_pixel = torch.unique(matrix).sort()[0].data.numpy()
    # matrix and classes of the processed images
    # img = torch.tensor(matrix, dtype=torch.float32)
    # img_2 = totensor(toimage(img).convert("RGB"))
    # img_2_dim = torch.sum(img_2, dim=0)

    os.makedirs(folder, exist_ok=True)

    classes, _, classes_to_models = get_mapping()
    # map class to a list of models

    toimage(original_img).save(os.path.join(folder, f"exp_real.png"))
    for i, pixel in enumerate(unique_pixel):
        # plane = original_img
        plane = torch.zeros_like(matrix, dtype=original_img.dtype).repeat(3, 1, 1)
        if image_classes[i] != 'background':
            plane = original_img
        else:
            for j in range(len(plane)):
                plane[j][torch.where(matrix == pixel)] = original_img[j][torch.where(matrix == pixel)]
                # plane[j][torch.where(plane[j] == 0)] = torch.mean(plane[j][torch.where(plane[j]!=0)])
        toimage(plane).save(os.path.join(folder, f"exp{i}.png"))

    # 调用cycleGAN
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
            fake_im = torchvision.transforms.functional.resize(fake_im, (matrix.shape[0], matrix.shape[1]), interpolation=2)
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
    new_images = torch.zeros(num, 3, *(matrix.shape))

    for i, (pixel, data, class_name) in enumerate(zip(unique_pixel, dataset, image_classes)):
        # get models
        buffer = 0
        if class_name == 'background':
            buffer = 1
            image_index = indexes.transpose(0, i)[0].flatten()

            real_im = toimage(util.tensor2im(data['A']))
            real = totensor(
                torchvision.transforms.functional.resize(real_im, (matrix.shape[0], matrix.shape[1]), interpolation=2))

            toimage(real).save(os.path.join(folder2, 'background_0.png'))
            for k in range(3):
                for idx in image_index:
                    new_images[idx][k][torch.where(matrix == pixel)] = real[k][torch.where(matrix == pixel)]

        for j, model_name in enumerate(classes_to_models[class_name]):
            model = models[model_name]

            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results

            fake_im = toimage(util.tensor2im(visuals['fake']))
            fake_im = torchvision.transforms.functional.resize(fake_im, (matrix.shape[0], matrix.shape[1]), interpolation=2)
            fake = totensor(fake_im)

            image_index = indexes.transpose(0, i)[j + buffer].flatten()
            for k in range(3):
                for idx in image_index:
                    new_images[idx][k][torch.where(matrix == pixel)] = fake[k][torch.where(matrix == pixel)]

            if class_name == 'background':
                toimage(fake).save(os.path.join(folder2, 'background_1.png'))
            else:
                toimage(fake).save(os.path.join(folder2, 'animal.png'))


    for i, new_image in enumerate(new_images):
        result = toimage(new_image)
        result.save(target_path+f'{i}.png')

    mix.mix(temp_folder2, target_path+'poisson')


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

    model_names = ['horse2zebra_pretrained', 'winter2summer_yosemite_pretrained']
    models = config_models(model_names, temp_folder)

    if args.image is not None:
        ISST(args.image, temp_folder, temp_folder2, os.path.join(target_folder, os.path.basename(args.image).split(".")[0] + '_fake'), models)
    else:
        assert args.folder is not None
        for i, input_path in enumerate(os.listdir(args.folder)):
            ISST(os.path.join(args.folder, input_path), temp_folder, temp_folder2, os.path.join(target_folder, os.path.basename(input_path).split(".")[0]  + '_fake'), models)
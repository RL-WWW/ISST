from PIL import Image
import torch
import os
import torchvision
from data import create_dataset
from models import create_model
from util import util
from util.visualizer import save_images
import torchvision.transforms as tfs
from options.test_options import TestOptions
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="C:/Users/ls/Desktop/exp/exp.png", help="The image we need to transfer")
    parser.add_argument("--real_image", default="C:/Users/ls/Desktop/exp/orig_exp.jpg", help="The original image")
    parser.add_argument("--temp_folder", default=None, help='Where to put the middle images')
    parser.add_argument("--target_path", default=None, help='Where to put the final result')
    return parser.parse_args()


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

    # real_img = totensor(Image.open("exp/exp_real.png"))

    target_path = args.target_path if args.target_path is not None else os.path.dirname(args.image)

    new_image = torch.zeros_like(img)

    for i, pixel in enumerate(torch.unique(img_2_dim)):
        plane = torch.zeros_like(img)
        for j in range(len(plane)):
            plane[j][torch.where(img_2_dim == pixel)] = original_img[j][torch.where(img_2_dim == pixel)]
        toimage(plane).save(os.path.join(folder, f"exp{i}.png"))

    # 调用cycleGAN
    model = "horse2zebra_pretrained"
    sys.argv = sys.argv[:1]
    sys.argv.append("--dataroot")
    sys.argv.append(folder)
    sys.argv.append("--name")
    sys.argv.append(model)
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

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    for i, (pixel, data) in enumerate(zip(torch.unique(img_2_dim), dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()

        toimage(visuals['real'].cpu().squeeze()).save("try_real.png")

        # A_path = data['A_paths'][0].split('.')
        # A_path[0] = A_path[0] + '_fake'
        # A_path = '.'.join(A_path)
        # toimage(visuals['fake'].squeeze()).save(A_path)
        # for i,pixel in enumerate(torch.unique(img_2_dim)):
        #     fake_im = Image.open(f"exp/exp{i}_fake.png")
        fake_im = toimage(util.tensor2im(visuals['fake']))
        fake = totensor(torchvision.transforms.functional.resize(fake_im, (img.shape[1], img.shape[2]), interpolation=2))
        for i in range(len(new_image)):
            new_image[i][torch.where(img_2_dim == pixel)] = fake[i][torch.where(img_2_dim == pixel)]

    #     new_image[torch.where(real_img == pixel)] = fake[torch.where(real_img == pixel)]
    #     print(torch.where(real_img == pixel))
    result = toimage(new_image)
    result.save(os.path.join(target_path, "result.png"))

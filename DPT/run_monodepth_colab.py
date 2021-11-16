"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from tqdm import tqdm
import numpy as np
from PIL import Image

#from util.misc import visualize_attention


def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    # img_names = glob.glob(os.path.join(input_path, "*"))
    # num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)
    #add para teste
    video_name = "./input/test-video.mp4"
    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 5 FPS
    os.system("mkdir input_images")
    os.system("mkdir output_images")
    count = 0
    ret = True

    print("start processing")

    while ret:
        ret, image = cap.read()
        if ret is False:
            break

        # get input
        img_names = 'img_' + str(count).zfill(4) + '.jpg'
        height, width, layers = image.shape
        resized_img = ''
        cropped_img = ''
        if height < width:
            nw = (width * 640) / height
            dim = (int(nw), 480)
            resized_img = cv2.resize(image, dim)
            w1 = int(nw / 2 - 320)
            w2 = int(nw / 2 + 320)
            cropped_img = resized_img[:, w1:w2]
        else:
            nh = (height * 640) / width
            dim = (640, int(nh))
            resized_img = cv2.resize(image, dim)
            h1 = int(nh / 2 - 240)
            h2 = int(nh / 2 + 240)
            cropped_img = resized_img[h1:h2, :]

        # img = image
        # args.kitti_crop = True
        # if args.kitti_crop is True:
        #     height, width, _ = img.shape
        #     top = height - 352
        #     left = (width - 1216) // 2
        #     img = img[top: top + 352, left: left + 1216, :]

        # img_input = transform({"image": img})["image"]
        cv2.imwrite(os.path.join('input_images', img_names), cropped_img)
        count += 1

    #get input
    img_names = glob.glob(os.path.join("input_images", "*"))
    num_images = len(img_names)

    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        img = util.io.read_image(img_name)
        img_input = transform({"image": img})["image"]
        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                    .squeeze()
                    .cpu()
                    .numpy()
            )

            if model_type == "dpt_hybrid_kitti":
                prediction *= 256

            if model_type == "dpt_hybrid_nyu":
                prediction *= 1000.0

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)

    print("finished")
    predict_dir("input_images/", "output_images/")
    os.system(
        "ffmpeg -i output_images/img_%04d.png -c:v libx264 -pix_fmt yuv444p output_images/out.mp4 && mplayer output_images/out.mp4")
    get_video()

def get_video():
    from IPython.display import HTML
    from base64 import b64encode
    video_path = "./output_images/out.mp4"

    mp4 = open(video_path, 'rb').read()
    decoded_vid = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f'<video width=100% controls><source src={decoded_vid} type="video/mp4"></video>')

@torch.no_grad()
def predict_dir(self, test_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    transform = ToTensor()
    all_files = glob.glob(os.path.join(test_dir, "*"))
    self.model.eval()
    for f in tqdm(all_files):
        image = np.asarray(Image.open(f), dtype='float32') / 255.
        image = transform(image).unsqueeze(0).to(self.device)

        centers, final = self.predict(image)
        # final = final.squeeze().cpu().numpy()

        final = (final * self.saving_factor).astype('uint16')
        basename = os.path.basename(f).split('.')[0]
        save_path = os.path.join(out_dir, basename + ".png")
        original = cv2.imread(f)
        img_arr = get_img_arr(original)
        output = display_single_image(final.squeeze(), img_arr) * 255

        cv2.imwrite(save_path, cv2.cvtColor(output.astype('float32'), cv2.COLOR_RGB2BGR))

def get_img_arr(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (640, 480))
    x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
    return x

def display_single_image(output, inputs=None, is_colormap=True):
    import matplotlib.pyplot as plt

    plasma = plt.get_cmap('plasma')

    imgs = []

    imgs.append(inputs)

    ##rescale output
    out_min = np.min(output)
    out_max = np.max(output)
    output = output - out_min
    outputs = output/out_max

    if is_colormap:
        pred_x = plasma(outputs)[:, :, :3]
        imgs.append(pred_x)

    img_set = np.hstack(imgs)

    return img_set
def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ToTensor(object):
    def __init__(self):
        self.normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(args.model_weights)
    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )

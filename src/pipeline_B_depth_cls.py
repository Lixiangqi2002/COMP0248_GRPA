from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
import pathlib
import random
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def visualize_depth_prediction(encoder: str = "vits", input_size: int = 518, examples_path: str = "CW2-Dataset/data/harvard_c5/hv_c5_1/image"):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(
        torch.load(f'log/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
    depth_anything = depth_anything.to(DEVICE).eval()

    examples_path = pathlib.Path(examples_path)
    images_lst: list[pathlib.Path] = list(examples_path.glob("*.jpg"))
    image_num: int = random.randint(0, len(images_lst) - 1)
    image = cv.imread(images_lst[image_num].__str__())[:, :, ::-1]

    depth_raw = depth_anything.infer_image(image, input_size)
    depth_raw = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min()) * 255.0
    depth_uint8 = depth_raw.astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth_image = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return image, depth_image

def plot_depth_prediction(image, depth_image):
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(image)
    plt.title("Input", fontsize=22)
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(depth_image)
    plt.title("Depth prediction", fontsize=22)
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    image, depth_image = visualize_depth_prediction()
    plot_depth_prediction(image, depth_image)
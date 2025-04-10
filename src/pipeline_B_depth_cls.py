from collections import defaultdict
import os
import sys
import cv2
from matplotlib import cm
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath('log/Depth_Anything_V2')) 
from depth_anything_v2.dpt import DepthAnythingV2
import pickle
from depth2cloud import generate_label_file, generate_point_cloud, read_intrinsics_from_txt
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from pipeline_A_pointnet2_train import main as train
from pipeline_A_pointnet2_test import main as test
import argparse

def find_closest_gt(pred_stem, gt_files):
    if "-" not in pred_stem:
        pred_id = int(pred_stem.split('_')[-1])
        min_diff = float('inf')
        closest = None
        for f in gt_files:
            gt_id = int(f.stem.split('_')[-1])
            diff = abs(gt_id - pred_id)
            if diff < min_diff:
                min_diff = diff
                closest = f
    else:
        pred_id = int(pred_stem.split('-')[-1])
        min_diff = float('inf')
        closest = None
        for f in gt_files:
            gt_id = int(f.stem.split('-')[-1])
            diff = abs(gt_id - pred_id)
            if diff < min_diff:
                min_diff = diff
                closest = f
    return closest


def evaluate_depth_metrics(pred_depth, gt_depth):
    mask = (gt_depth > 0) & np.isfinite(gt_depth) & np.isfinite(pred_depth)

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    if len(pred) < 10:
        return None  # too few valid pixels

    abs_rel = np.mean(np.abs(pred - gt) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred + 1e-6) - np.log(gt + 1e-6)) ** 2))

    thresh = np.maximum(pred / gt, gt / pred)
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)

    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "RMSE(log)": rmse_log,
        "δ<1.25": delta1,
        "δ<1.25^2": delta2,
        "δ<1.25^3": delta3
    }



def eval_depth_all(pred_dir, gt_dir, verbose=False):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.npy"))
    results = []
    gt_files = list(gt_dir.glob("*.npy"))
    for pred_file in tqdm(pred_files, desc="Evaluating depth predictions"):
        pred = np.load(pred_file)
        gt_file = find_closest_gt(pred_file.stem, gt_files)
        if gt_file is None:
            continue
        gt_xyz = np.load(gt_file)  # (N, 3)

        try:
            gt_depth = gt_xyz.astype(np.float32)
        except:
            if verbose:
                print(f"[!] GT shape mismatch for {gt_file}")
            continue

        # Resize pred to match the shape of gt_depth
        if pred.shape != gt_depth.shape:
            pred = cv2.resize(pred, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        metrics = evaluate_depth_metrics(pred, gt_depth)
        results.append(metrics)

        # if verbose:
        #     print(f"[{pred_file.name}]")
        #     for k, v in metrics.items():
        #         print(f"  {k}: {v:.4f}")

    if not results:
        print("[!] No valid evaluations done.")
        return

   
    keys = results[0].keys()
    avg_metrics = {k: np.mean([r[k] for r in results]) for k in keys}

    print("\n === Evaluation Summary ===")
    for k, v in avg_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    return avg_metrics


def predict_and_save_depths(
    rgb_dir: str,
    save_dir: str,
    encoder: str = "vits",
    input_size: int = 518,
    vis_range=(0.1, 50.0),
    cmap: str = "gray",
    gt_base_Path: str = "data/dataset/point_clouds",
    epsilon: int = 200,
):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model config
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Load model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'log/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth'
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)
    depth_anything = depth_anything.to(DEVICE).eval()

    rgb_dir = Path(rgb_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mode = "train" if "train" in str(rgb_dir) else "test"
    gt_base = Path(gt_base_Path + f"/{mode}")
    # print(f"GT base path: {gt_base}")
    images = sorted(rgb_dir.glob("*.jpg"))
    vmin, vmax = vis_range

    for img_path in tqdm(images, desc=f"Inferencing {rgb_dir.name}"):
        image = cv2.imread(str(img_path))[:, :, ::-1]
        pred = depth_anything.infer_image(image, input_size)  # inverse depth (relative)

        gt_npy_path = None
        for gt_file in gt_base.glob("*.npy"):
            id_img = str(img_path.stem.split('_')[-1])
            if gt_file.name.endswith(id_img+"_full.npy"):
                gt_npy_path = gt_file
                break
            

        if gt_npy_path is None:
            print(f"[!] GT not found for: {img_path.stem}, skipping")
            continue
        gt = np.load(gt_npy_path)

        gt_median = np.median(-gt[:, 2]) 
        print(f"GT median: {gt_median}") 
        # pred_depth = 1.0 / (pred + epsilon)
        pred_depth = pred
        pred_median = np.median(pred_depth)
        # print(f"Pred median: {pred_median}")
        scale = gt_median / pred_median
        depth_map = pred_depth * scale

        npy_path = save_dir / f"{img_path.stem}.npy"
        np.save(npy_path, depth_map)

        # clipped = np.clip(depth_map, vmin, vmax)
        # norm = (clipped - vmin) / (vmax - vmin)
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        vis_path = save_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(vis_path), depth_vis)
        # break
    print(f"Done. Saved predictions + PNGs to: {save_dir}")



def generate_point_cloud_depth_prediction():
    base_path = Path("data/CW2-Dataset/data")
    output_path = Path("data/dataset/depth_prediction_point_clouds")
    fx, fy, cx, cy = 570.3422, 570.3422, 320, 240
    # fx, fy, cx, cy = 518.0, 518.0, 320, 240

    for mode in ["train", "test"]: #
        image_dir = Path(f"data/dataset/depth_img/{mode}/image")
        depth_dir = Path(output_path) / mode / "depth"
        # label_dir = Path(output_path) / mode / "labels"
        # label_dir.mkdir(parents=True, exist_ok=True)
        out_prefix_base = output_path / mode

        image_files = sorted(image_dir.glob("*.jpg"))
        depth_files = sorted(depth_dir.glob("*.png"))
        scene_counters = defaultdict(int)
        for i, (img_file, depth_file) in enumerate(tqdm(zip(image_files, depth_files), total=len(image_files), desc=f"Generating PCL ({mode})")):
       
            split_list = ['harvard_c5', 'harvard_c6', 'harvard_c11', 'harvard_tea_2',
              'mit_32_d507', 'mit_76_459', 'mit_76_studyroom', 'mit_gym_z_squash', 'mit_lab_hj']

            key = img_file.stem  # e.g., harvard_c5_hv_c5_1_000001-0000023574
            prefix = "_".join(key.split("_")[:-1])  # harvard_c5_hv_c5_1
            # print(prefix)
            split = None
            scene = None
            for s in split_list:
                if prefix.startswith(s):
                    split = s
                    scene = prefix[len(s)+1:]  # remove split + underscore
                    break
            # if split == 'mit_32_d507' or split =='mit_76_459' or split == 'mit_76_studyroom' or split == 'mit_gym_z_squash' :#or split == 'mit_lab_hj':
            #     continue
            # print(split, scene)
            if split is None or scene is None:
                raise ValueError(f"Cannot determine split/scene from filename prefix: {prefix}")

            label_path = Path(base_path) / split / scene / "labels" / "tabletop_labels.dat"
            # print(label_path)
            scene_key = f"{split}_{scene}"
            scene_frame_idx = scene_counters[scene_key]
            scene_counters[scene_key] += 1
            if label_path.exists():
                with open(label_path, 'rb') as f:
                    labels = pickle.load(f)

                if scene_frame_idx < len(labels) and labels[scene_frame_idx] is not None:
                    polygon_list = labels[scene_frame_idx]
                else:
                    polygon_list = []
                # print(f"[{scene_key}] Using scene_frame_idx = {scene_frame_idx}, total labels = {len(labels)}")

            else:
                polygon_list = []

            out_prefix = out_prefix_base / "point_clouds" 
            out_prefix.mkdir(parents=True, exist_ok=True)
            # print(out_prefix)

            # # visualize the labels on the rgb images
            # img = plt.imread(img_file)
            # plt.imshow(img)
            # for polygon in polygon_list:
            #     plt.plot(polygon[0]+polygon[0][0:1],polygon[1]+polygon[1][0:1],'r')
            #     # plt.savefig(str(out_prefix) + "/" + f"{img_file.stem}_label.png", bbox_inches='tight', pad_inches=0)
            # plt.axis('off')
            # plt.show()    

            generate_point_cloud(
                rgb_file=str(img_file),
                depth_file=str(depth_file),
                polygon_list=polygon_list,
                fx=fx, fy=fy, cx=cx, cy=cy,
                output_prefix_labels=str(out_prefix) + "/" + key
            )


def process_all_single_folder(image_root, depth_root, intrinsic_path, output_dir):
 
    dir_name = 'test'

    if not os.path.exists(intrinsic_path):
        raise FileNotFoundError(f"intrinsics.txt not found in {intrinsic_path}")

    fx, fy, cx, cy = read_intrinsics_from_txt(intrinsic_path)
    print(f"Intrinsic parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    image_files = sorted(os.listdir(image_root))
    depth_files = sorted(os.listdir(depth_root))
    image_basenames = {os.path.splitext(f)[0]: f for f in image_files}
    depth_basenames = {os.path.splitext(f)[0]: f for f in depth_files}
    common_keys = sorted(set(image_basenames.keys()) & set(depth_basenames.keys()))

    print(f"Found {len(common_keys)} matched image/depth pairs.")

    for key in common_keys:
        img_file = image_basenames[key]
        depth_file = depth_basenames[key]

        print(f"Processing {img_file} and {depth_file}")
        rgb_path = os.path.join(image_root, img_file)
        d_path = os.path.join(depth_root, depth_file)

        filename_no_ext = key
        out_prefix = os.path.join(output_dir, dir_name, f"{filename_no_ext}")

        generate_point_cloud(rgb_path, d_path, None, fx, fy, cx, cy, out_prefix)


def convert_depth_png_to_npy(input_dir, output_dir, divisor=1000.0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_pngs = sorted(input_dir.glob("*.png"))
    print(f"Converting {len(depth_pngs)} PNG depth files...")

    for png_path in tqdm(depth_pngs):
        depth_mm = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        depth_m = depth_mm.astype(np.float32) / divisor

        npy_path = output_dir / f"{png_path.stem}.npy"
        np.save(npy_path, depth_m)

    print(f"Done. Saved .npy files to: {output_dir}")  


def parse_args_train():
    parser = argparse.ArgumentParser('PointNet2 Binary Classification in Pipeline B')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_point', type=int, default=2048, help='Point number per sample')
    parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_B', help='Log directory name')
    parser.add_argument('--resume', default=False,  action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--train_label_path', type=str, default='data/dataset/depth_prediction_point_clouds/train/point_clouds/train_labels.txt', help='Path to training label file')
    return parser.parse_args()


def parse_args_test():
    if dataset == "CW2":
        parser = argparse.ArgumentParser('PointNet2 Binary Classification Testing in Pipeline B')
        parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_B', help='Experiment log directory')
        parser.add_argument('--test_label_path', type=str, default='data/dataset/point_clouds/test/test_labels.txt', help='Path to test label file')
    else:
        parser = argparse.ArgumentParser('PointNet2 Binary Classification Testing in Pipeline B RealSense')
        parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_B_realsense', help='Experiment log directory')
        parser.add_argument('--test_label_path', type=str, default='data/dataset/realsense_point_clouds/test_labels.txt', help='Path to test label file')

    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    
    return parser.parse_args() 
    

if __name__ == '__main__':
    dataset = "RealSense"  # "RealSense" or "CW2"
    if dataset == "CW2":
        ### CW2 data ###
        predict_and_save_depths(
            rgb_dir="data/dataset/depth_img/train/image",
            save_dir="data/dataset/depth_prediction_point_clouds/train/depth",
            encoder="vitl"
        )

        predict_and_save_depths(
            rgb_dir="data/dataset/depth_img/test/image",
            save_dir="data/dataset/depth_prediction_point_clouds/test/depth",
            encoder="vitl"
        )

        convert_depth_png_to_npy(
            input_dir="data/dataset/depth_img/train/depth",
            output_dir="data/dataset/depth_prediction_point_clouds/train/depth_gt"
        )

        convert_depth_png_to_npy(
            input_dir="data/dataset/depth_img/test/depth",
            output_dir="data/dataset/depth_prediction_point_clouds/test/depth_gt"
        )

        eval_depth_all(
            pred_dir="data/dataset/depth_prediction_point_clouds/train/depth",
            gt_dir="data/dataset/depth_prediction_point_clouds/train/depth_gt",
            verbose=True
        )
        eval_depth_all(
            pred_dir="data/dataset/depth_prediction_point_clouds/test/depth",
            gt_dir="data/dataset/depth_prediction_point_clouds/test/depth_gt",
            verbose=True
        )

        generate_point_cloud_depth_prediction()
        generate_label_file("data/dataset/depth_prediction_point_clouds/train/point_clouds", "data/dataset/depth_prediction_point_clouds/train/point_clouds/train_labels.txt")
        generate_label_file("data/dataset/depth_prediction_point_clouds/test/point_clouds", "data/dataset/depth_prediction_point_clouds/test/point_clouds/test_labels.txt")


        # training point cloud classification model as pipeline A
        # but use the pipeline B point cloud dataset
        args_train = parse_args_train()
        train(args=args_train)

        args_test = parse_args_test()
        test(args=args_test)
    else:
        # ### RealSense data ### 
        predict_and_save_depths(
            rgb_dir="data/dataset/realsense_depth_img/test/image",
            save_dir="data/dataset/realsense_depth_prediction_point_clouds/test/depth",
            encoder="vitl",
            gt_base_Path="data/dataset/realsense_point_clouds",
            epsilon=100
        )
        convert_depth_png_to_npy(
            input_dir="data/dataset/realsense_depth_img/test/depth",
            output_dir="data/dataset/realsense_depth_prediction_point_clouds/test/depth_gt"
        )
        eval_depth_all(
            pred_dir="data/dataset/realsense_depth_prediction_point_clouds/test/depth",
            gt_dir="data/dataset/realsense_depth_prediction_point_clouds/test/depth_gt",
            verbose=True
        )
        process_all_single_folder(image_root="data/dataset/realsense_depth_img/test/image",
                                  depth_root="data/dataset/realsense_depth_prediction_point_clouds/test/depth",
                                  intrinsic_path="data/dataset/realsense_point_clouds/intrinsics.txt",
                                  output_dir="data/dataset/realsense_depth_prediction_point_clouds/test/point_clouds")
        args_test = parse_args_test()
        test(args=args_test)
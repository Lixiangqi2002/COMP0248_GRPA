from pathlib import Path
import pickle
import shutil
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image
import open3d as o3d
import cv2
import os


class point_cloud_generator():

    def __init__(self, rgb_file, depth_file, output_prefix, fx, fy, cx, cy, scalingfactor=1000, do_inpaint=False):
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.output_prefix = output_prefix  # e.g. "data/frame00001"
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scalingfactor = scalingfactor
        self.rgb = Image.open(rgb_file)
        rgb_width, rgb_height = self.rgb.size
        if depth_file.endswith(".npy"):
            self.depth = np.load(depth_file)#.T  # shape: (H, W)
            self.depth_type = "npy"
        else:
            depth_raw = Image.open(depth_file).convert('I')
            if depth_raw.size != self.rgb.size:
                print(f"[Resize] Depth image size {depth_raw.size} -> {self.rgb.size}")
                depth_raw = depth_raw.resize((rgb_width, rgb_height), resample=Image.NEAREST)
            self.depth = np.asarray(depth_raw)
            self.depth_type = "png"
        # self.depth = Image.open(depth_file).convert('I')  # uint16
        self.width = self.rgb.size[0]
        self.height = self.rgb.size[1]
        if do_inpaint:
            self.depth = self.inpaint_depth(self.depth)
            # self.visualize_depth()

    def inpaint_depth(self, depth):
        if self.depth_type != "png":
            print("[Inpaint] Only support .png depth, skip inpaint.")
            return depth

        mask = (depth == 0).astype(np.uint8)
        depth_min = np.min(depth[depth > 0])
        depth_max = np.max(depth)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        inpainted = cv2.inpaint(depth_normalized, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        restored = inpainted.astype(np.float32) / 255 * (depth_max - depth_min) + depth_min
        # restored = inpainted.astype(np.float32) / 255 * np.max(depth)
        print(f"[Debug] Depth min={np.min(restored)}, max={np.max(restored)}, mean={np.mean(restored)}")

        return restored.astype(np.uint16)

    def visualize_depth(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.depth, cmap='gray')
        plt.title("Inpainted Depth")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        rgb_vis = np.asarray(self.rgb)
        plt.imshow(rgb_vis)
        plt.title("RGB Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def calculate(self, polygon_mask=None):
        t1 = time.time()
        depth = np.asarray(self.depth).T  # (H, W)
        # Z = depth / self.scalingfactor
        if self.depth_type == "npy":
            Z = np.asarray(self.depth)  # m
            img = np.array(self.rgb)#.transpose(1, 0, 2)
            u = np.arange(self.width)
            v = np.arange(self.height)
            u_grid, v_grid = np.meshgrid(u, v, indexing='xy')

        else:
            Z = depth / self.scalingfactor  # PNG: uint16 mm ➜ m
            img = np.array(self.rgb).transpose(1, 0, 2)  # (W, H, 3)

            u = np.arange(self.width)
            v = np.arange(self.height)
            u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

        X = ((u_grid - self.cx) * Z) / self.fx
        Y = ((v_grid - self.cy) * Z) / self.fy


        valid = (Z > 0)

        X_full = X[valid]
        Y_full = Y[valid]
        Z_full = Z[valid]
        img_full = img[valid]

        df_full = np.zeros((6, len(X_full)))
        df_full[0] = X_full
        df_full[1] = -Y_full
        df_full[2] = -Z_full
        df_full[3] = img_full[:, 0]
        df_full[4] = img_full[:, 1]
        df_full[5] = img_full[:, 2]
        self.df_full = df_full

        if polygon_mask is not None:
            if self.depth_type == "png":
                polygon_mask = polygon_mask.T
            valid_mask = valid & (polygon_mask > 0)
        else:
            valid_mask = valid

        X_sel = X[valid_mask]
        Y_sel = Y[valid_mask]
        Z_sel = Z[valid_mask]
        img_sel = img[valid_mask]

        df_sel = np.zeros((6, len(X_sel)))
        df_sel[0] = X_sel
        df_sel[1] = -Y_sel
        df_sel[2] = -Z_sel
        df_sel[3] = img_sel[:, 0]
        df_sel[4] = img_sel[:, 1]
        df_sel[5] = img_sel[:, 2]
        self.df_masked = df_sel

        t2 = time.time()
        # print(f'3D point cloud (full & mask) computed in {t2 - t1:.2f}s')

    def write_ply(self):
        os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)

        for tag, df in zip(['full', 'mask'], [self.df_full, self.df_masked]):
            # if tag == 'mask':
            #     break
            out_file = f"{self.output_prefix}_{tag}.ply"

            float_formatter = lambda x: "%.4f" % x
            points = []
            for i in df.T:
                points.append("{} {} {} {} {} {} 0\n".format(
                    float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                    int(i[3]), int(i[4]), int(i[5])
                ))

            with open(out_file, "w") as file:
                file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
    ''' % (len(points), "".join(points)))
                # print(f"Saved .ply: {out_file}")

    def save_npy(self):
        os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)
        np.save(f"{self.output_prefix}_full.npy", self.df_full.T[:, :3])   
        np.save(f"{self.output_prefix}_mask.npy", self.df_masked.T[:, :3])
        # print(f"Saved .npy: {self.output_prefix}_full.npy & _mask.npy")

    def show_point_cloud(self, tag='mask'):
        ply_file = f"{self.output_prefix}_{tag}.ply"
        pcd = o3d.io.read_point_cloud(ply_file)
        o3d.visualization.draw_geometries([pcd])


def create_polygon_mask(polygon_list, height, width):
    """Generate binary mask (H, W) from polygon list"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygon_list:
        pts = np.array(list(zip(polygon[0], polygon[1])), dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
    
    return mask

def read_intrinsics_from_txt(file_path):
    """Read intrinsics from text file and return Open3D PinholeCameraIntrinsic"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    fx, _, cx = map(float, lines[0].split())
    _, fy, cy = map(float, lines[1].split())

    return fx, fy, cx, cy



def generate_point_cloud(rgb_file, depth_file, polygon_list, fx, fy, cx, cy, output_prefix_labels, data="CW2"):
    if not polygon_list: 
        mask = np.zeros((480, 640), dtype=np.uint8)
    else:
        mask = create_polygon_mask(polygon_list, height=480, width=640)
    if data == "CW2":
        pcg = point_cloud_generator(rgb_file, depth_file, output_prefix_labels, fx, fy, cx, cy, do_inpaint=False)
        pcg.calculate(polygon_mask=mask)
    elif data == "Realsense":
        pcg = point_cloud_generator(rgb_file, depth_file, output_prefix_labels, fx, fy, cx, cy, do_inpaint=True)
        pcg.calculate(polygon_mask=None)
    pcg.write_ply()
    pcg.save_npy()
    # pcg.show_point_cloud(tag='mask')
    # pcg.show_point_cloud(tag='full')


def process_all(base_dir, output_dir):
    """Automatically process all train/test sequences"""
    train_splits = ['mit_32_d507', 'mit_76_459', 'mit_76_studyroom', 'mit_gym_z_squash', 'mit_lab_hj']
    test_splits = ['harvard_c5', 'harvard_c6', 'harvard_c11', 'harvard_tea_2']

    all_splits = train_splits + test_splits
    dir_name = ''
    for split in all_splits:
        print(f"Processing {split}...")
        if split[0:3] == 'mit':
            dir_name = 'train'
        else:
            dir_name = 'test'
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue

        subfolders = [os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for scene in subfolders:
            image_dir = os.path.join(scene, 'image')
            depth_dir = os.path.join(scene, 'depth') if os.path.exists(os.path.join(scene, 'depth')) else os.path.join(scene, 'depthTSDF')
            label_path = os.path.join(scene, 'labels', 'tabletop_labels.dat')
            intrinsic_path = os.path.join(scene, 'intrinsics.txt')
            print(image_dir)
            print(depth_dir)
            print(label_path)
            print(intrinsic_path)
            fx, fy, cx, cy = read_intrinsics_from_txt(intrinsic_path)

            if not (os.path.exists(image_dir) and os.path.exists(depth_dir) and os.path.exists(intrinsic_path)):
                continue
            if not os.path.exists(label_path):
                tabletop_labels = np.array([None] * len(os.listdir(image_dir)))
            else:
                with open(label_path, 'rb') as label_file:
                    tabletop_labels = pickle.load(label_file)
                    label_file.close()  

            # Process all frames in the scene
            image_files = sorted(os.listdir(image_dir))
            depth_files = sorted(os.listdir(depth_dir))
            print(len(image_files))
            for polygon_list, img_file, depth_file in zip(tabletop_labels, image_files, depth_files):
                # print(img_file)
                # print(depth_file)
                rgb_path = os.path.join(image_dir, img_file)
                d_path = os.path.join(depth_dir, depth_file)
                filename_no_ext = os.path.splitext(img_file)[0]
                out_prefix = os.path.join(output_dir+f"/{dir_name}", f"{split}_{os.path.basename(scene)}_{filename_no_ext}")
                generate_point_cloud(rgb_path, d_path, polygon_list, fx, fy, cx, cy, out_prefix, data="CW2")
                # # visualize the labels on the rgb images
                # img = plt.imread(rgb_path)
                # plt.imshow(img)
                # for polygon in polygon_list:
                #     plt.plot(polygon[0]+polygon[0][0:1],polygon[1]+polygon[1][0:1],'r')
                # plt.axis('off')
                # plt.show()    
  
        #         break
        #     break  
        # break    


def process_all_single_folder(image_root, depth_root, intrinsic_path, output_dir):
 
    dir_name = 'test'

    if not os.path.exists(intrinsic_path):
        raise FileNotFoundError(f"intrinsics.txt not found in {intrinsic_path}")

    fx, fy, cx, cy = read_intrinsics_from_txt(intrinsic_path)

    scenes = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]

    for scene in scenes:
        print(f"Processing scene: {scene}")

        image_dir = os.path.join(image_root, scene)
        depth_dir = os.path.join(depth_root, scene)

        if not os.path.exists(image_dir) or not os.path.exists(depth_dir):
            print(f"Skipping scene {scene} due to missing image or depth folder.")
            continue

        image_files = sorted(os.listdir(image_dir))
        depth_files = sorted(os.listdir(depth_dir))

        if len(image_files) != len(depth_files):
            print(f"Warning: Mismatched image/depth count in scene {scene}")

        tabletop_labels = [None] * len(image_files)

        for polygon_list, img_file, depth_file in zip(tabletop_labels, image_files, depth_files):
            rgb_path = os.path.join(image_dir, img_file)
            d_path = os.path.join(depth_dir, depth_file)
            filename_no_ext = os.path.splitext(img_file)[0].replace("rgb_", "")

            out_prefix = os.path.join(output_dir, dir_name, f"{scene}_{filename_no_ext}")
            generate_point_cloud(rgb_path, d_path, polygon_list, fx, fy, cx, cy, out_prefix, data ="Realsense")



def generate_label_file(split_dir, output_file, data="CW2"):
    entries = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if data == "CW2":
                if file.endswith("_mask.npy"):
                    # print(file)
                    file_path = os.path.join(root, file)
                    pc = np.load(file_path)
                    label = 0 if pc.shape[0] == 0 else 1
                    file_path = file_path.replace('_mask.npy', '_full.npy')
                    entries.append(f"{file_path} {label}\n")
            elif data == "Realsense":
                label = 1
                if "no_table" in file:
                    label = 0
                file_path = os.path.join(root, file)
                # pc = np.load(file_path, allow_pickle=True)
                entries.append(f"{file_path} {label}\n")
    with open(output_file, "w") as f:
        f.writelines(entries)
    print(f"Saved {len(entries)} entries to {output_file}")


def copy_scene_images(scene_list, out_img_dir, out_depth_dir, src_root):
    for scene in scene_list:
        scene_dir = src_root / scene
        if not scene_dir.exists():
            print(f"[Warning] Scene not found: {scene_dir}")
            continue
        for subscene in sorted(scene_dir.iterdir()):
            if not subscene.is_dir(): continue
            image_dir = subscene / "image"
            depth_dir = subscene / "depthTSDF" if (subscene / "depthTSDF").exists() else subscene / "depth"

            if not image_dir.exists() or not depth_dir.exists():
                print(f"[Skip] Missing image/depth folder in {subscene}")
                continue

            img_files = sorted(os.listdir(image_dir))
            depth_files = sorted(os.listdir(depth_dir))

            # Ensure matching count
            assert len(img_files) == len(depth_files), f"Mismatch: {len(img_files)} RGBs vs {len(depth_files)} depth maps in {subscene}"

            for img_fname, depth_fname in zip(img_files, depth_files):
                # print(f"RGB: {img_fname} | Depth: {depth_fname}")
                src_img = image_dir / img_fname
                src_depth = depth_dir / depth_fname

                out_img = out_img_dir / f"{scene}_{subscene.name}_{img_fname}"
                out_depth = out_depth_dir / f"{scene}_{subscene.name}_{depth_fname}"

                shutil.copy(src_img, out_img)
                shutil.copy(src_depth, out_depth)


def organize_rgbd_dataset(src_root, train_scenes,test_scenes,dst_root: str = "data/dataset/depth_img"):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    out_train_img = dst_root / "train" / "image"
    out_train_depth = dst_root / "train" / "depth"
    out_test_img = dst_root / "test" / "image"
    out_test_depth = dst_root / "test" / "depth"
    
    for p in [out_train_img, out_train_depth, out_test_img, out_test_depth]:
        p.mkdir(parents=True, exist_ok=True)

    copy_scene_images(train_scenes, out_train_img, out_train_depth,src_root)
    copy_scene_images(test_scenes, out_test_img, out_test_depth, src_root)

    print("Done: RGB-D dataset organized into:")
    print(f"  {out_train_img.parent} and {out_test_img.parent}")



def extract_number(filename):
    import re
    match = re.search(r'\d+', filename)
    return match.group() if match else None



def copy_flat_scene_images(scene_names, out_img_dir, out_depth_dir, src_root):
    for scene in scene_names:
        image_dir = Path(src_root) / 'image' / scene
        depth_dir = Path(src_root) / 'depth' / scene

        if not image_dir.exists() or not depth_dir.exists():
            print(f"[Warning] Missing: {image_dir} or {depth_dir}")
            continue
   
        img_files = sorted(os.listdir(image_dir))
        depth_files = sorted(os.listdir(depth_dir))

        img_basenames = {extract_number(f): f for f in img_files}
        depth_basenames = {extract_number(f): f for f in depth_files}
        common = sorted(set(img_basenames.keys()) & set(depth_basenames.keys()))

        print(f"Processing scene: {scene}")
        print(f"  {len(common)} common images found")
        for name in common:
            src_img = image_dir / img_basenames[name]
            src_depth = depth_dir / depth_basenames[name]

            out_img = out_img_dir / f"{scene}_{img_basenames[name]}"
            out_depth = out_depth_dir / f"{scene}_{depth_basenames[name]}"

            shutil.copy(src_img, out_img)
            shutil.copy(src_depth, out_depth)



def organize_rgbd_dataset_single_dataset(src_root, test_scenes, dst_root: str = "data/dataset/depth_img"):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    out_test_img = dst_root / "test" / "image"
    out_test_depth = dst_root / "test" / "depth"
    
    for p in [out_test_img, out_test_depth]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"src_root: {src_root}")
    print(f"out_test_img: {out_test_img}")
    copy_flat_scene_images(test_scenes, out_test_img, out_test_depth, src_root)

    print(f"Done: RGB-D dataset organized into: {out_test_img.parent}")



if __name__ == '__main__':
    data = "Realsense" # 'CW2' or "Realsense"
    if data == 'CW2':
        ### CW2 dataset ###
        base_path = "data/CW2-Dataset/data"

        # point cloud dataset orgnization
        output_path = "data/dataset/point_clouds"
        process_all(base_path, output_path)
    
        generate_label_file("data/dataset/point_clouds/train", "data/dataset/point_clouds/train/train_labels.txt")
        generate_label_file("data/dataset/point_clouds/test", "data/dataset/point_clouds/test/test_labels.txt")

        # RGB-D dataset organization
        output_image_path = "data/dataset/depth_img"
        train_scenes = [
            'mit_32_d507', 'mit_76_459', 'mit_76_studyroom',
            'mit_gym_z_squash', 'mit_lab_hj'
        ]
        test_scenes = ['harvard_c5', 'harvard_c6', 'harvard_c11', 'harvard_tea_2']

        organize_rgbd_dataset(
            src_root=base_path,
            train_scenes=train_scenes,
            test_scenes=test_scenes,
            dst_root=output_image_path
        )

    elif data == 'Realsense':
        ### Realsense dataset ###
        
        # point cloud dataset organization
        base_path_realsense = "data/realsense"
        output_path_realsense = "data/dataset/realsense_point_clouds"
        image_root = os.path.join(base_path_realsense, 'image')
        depth_root = os.path.join(base_path_realsense, 'depth')
        intrinsic_path = os.path.join(base_path_realsense, 'intrinsics.txt')
        # generate npy & ply
        # process_all_single_folder(image_root, depth_root, intrinsic_path, output_path_realsense)
        # genearte label files
        generate_label_file("data/dataset/realsense_point_clouds/test", "data/dataset/realsense_point_clouds/test/test_labels.txt",data="Realsense")

        scene_names = os.listdir(os.path.join(base_path_realsense+'/image'))
        print(scene_names)
        organize_rgbd_dataset_single_dataset(
            src_root=base_path_realsense,
            test_scenes=scene_names,
            dst_root='data/dataset/realsense_depth_img'
        )
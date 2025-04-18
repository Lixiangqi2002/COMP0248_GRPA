from pathlib import Path
import pickle
import shutil
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image
import open3d as o3d
import cv2
import os
from sklearn.linear_model import RANSACRegressor

from scipy import stats

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
            # self.adjust_rgb_brightness_contrast(alpha=0.3, beta=100)
            self.depth = self.inpaint_depth(self.depth)
            depth_name = os.path.basename(depth_file)
            depth_name_no_ext = os.path.splitext(depth_name)[0]
            filename ="data/dataset/realsense_depth_img_C/test/depth_inpainted/" + depth_name_no_ext + "_inpainted.png"
            print(f"[Inpaint] Save inpainted depth to {filename}")
            self.visualize_depth(filename)
    

    def inpaint_depth(self, depth):
        if self.depth_type != "png":
            print("[Inpaint] Only support .png depth, skip inpaint.")
            return depth

        mask = (depth == 0).astype(np.uint8)
        depth_min = np.min(depth[depth > 0])
        depth_max = np.max(depth)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        inpainted = cv2.inpaint(depth_normalized, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

        restored = inpainted.astype(np.float32) / 255 * (depth_max - depth_min) + depth_min
   
        smoothed = cv2.bilateralFilter(restored.astype(np.float32), d=7, sigmaColor=75, sigmaSpace=75)

        restored[mask == 1] = smoothed[mask == 1]
        print(f"[Debug] Depth min={np.min(restored)}, max={np.max(restored)}, mean={np.mean(restored)}")
        # restored = fit_plane_ransac(restored, mask)
        return restored.astype(np.uint16)


    def visualize_depth(self, out_filename):
        plt.figure(figsize=(12, 5))
        plt.imshow(self.depth, cmap='gray')
        plt.axis("off")
        plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def calculate(self, polygon_mask=None):
        """Calculate 3D point cloud from RGB and depth images"""
        t1 = time.time()
        depth = np.asarray(self.depth).T  # (H, W)
        Z = depth / self.scalingfactor

        # compute 3D coordinates of each pixel
        # u, v are the pixel coordinates in the image
        # Z is the depth value at each pixel
        u = np.arange(self.width)
        v = np.arange(self.height)
        u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

        # X, Y, Z are the 3D coordinates in camera space
        X = ((u_grid - self.cx) * Z) / self.fx
        Y = ((v_grid - self.cy) * Z) / self.fy
        X *= 300 
        Y *= 300
        Z *= 300
        # Convert RGB image to numpy array and transpose to (H, W, 3)
        img = np.array( self.rgb).transpose(1, 0, 2)  # (W, H, 3)

        # check if the depth value is valid (greater than 0)
        valid = (Z > 0)  
        # print(f"[Depth Stats] valid pixel count: {len(valid)}, min: {np.min(valid)}, max: {np.max(valid)}")


        X_full = X[valid]
        Y_full = Y[valid]
        Z_full = Z[valid]
        img_full = img[valid]

        # df_full means full point cloud
        # df_masked means point cloud with mask
        df_full = np.zeros((6, len(X_full)))
        df_full[0] = X_full
        df_full[1] = -Y_full
        df_full[2] = -Z_full
        df_full[3] = img_full[:, 0]
        df_full[4] = img_full[:, 1]
        df_full[5] = img_full[:, 2]
        self.df_full = df_full

        # apply polygon mask if provided
        if polygon_mask is not None:
            polygon_mask = polygon_mask.T  # match (W, H)
            valid_mask = valid & (polygon_mask > 0)
        else:
            valid_mask = valid

        # apply mask to X, Y, Z, img
        # X_sel, Y_sel, Z_sel, img_sel are the selected points after applying mask
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

        # write full point cloud to .ply file
        # tag, df is the name of the file, df is the data frame
        # e.g. tag: full, df: self.df_full
        # e.g. tag: mask, df: self.df_masked
        for tag, df in zip(['full', 'mask'], [self.df_full, self.df_masked]):
            out_file = f"{self.output_prefix}_{tag}.ply"

            float_formatter = lambda x: "%.4f" % x      # 4 decimal places, e.g. 0.1234
            points = []
            for i in df.T:
                points.append("{} {} {} {} {} {} 0\n".format(
                    float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                    int(i[3]), int(i[4]), int(i[5])
                ))

            # write to .ply file
            # structure:
            # ply: used to indicate the file is in PLY format
            # format ascii 1.0: means ASCII format
            # element vertex N: number of points
            # property float x: x coordinate
            # property float y: y coordinate
            # property float z: z coordinate
            # property uchar red: red color
            # property uchar green: green color
            # property uchar blue: blue color
            # property uchar alpha: alpha color, meaning transparency
            # end_header:
            with open(out_file, "w") as file:
                file.write(
                            '''ply
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
                            ''' % (len(points), "".join(points))
                            )
        # print(f"Saved .ply: {out_file}")

    def save_npy(self):
        """Save point cloud data to .npy file"""
        """Output format:
        full.npy: (N, 9): [x, y, z, r, g, b, norm_x, norm_y, norm_z]
        label.npy: (N, 1): [label]"""
        # here you need to save a 9 dimensional array for full point cloud
        # and a 3 dimensional array for masked point cloud
        # e.g. full.npy: (N, 9), labellde.npy: (N, 3)

        os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)
        
        pc = self.df_full.T[:, :6]          # shape (N, 6): [x, y, z, r, g, b]
        mask_pc = self.df_masked.T[:, :3]   # shape (M, 3): [x, y, z]
        
        pc_xyz = pc[:, :3]
        rgb = pc[:, 3:6]
        # normalize the point cloud coordinates
        max_xyz = np.max(pc_xyz, axis=0)
        norm_xyz = pc_xyz / max_xyz  # (N, 3)

        # concatenate: xyz, rgb, norm_xyz
        full = np.concatenate([pc_xyz, rgb, norm_xyz], axis=1)  # shape (N, 9)

        # print shapes for debugging
        print(f"full shape: {full.shape}, mask shape: {mask_pc.shape}")

        # compute label
        pc_rounded = np.round(pc_xyz, 3)
        mask_rounded = np.round(mask_pc, 3)
        mask_set = set(map(tuple, mask_rounded))
        labels = np.array([1 if tuple(p) in mask_set else 0 for p in pc_rounded], dtype=np.int64)

        # save full point cloud and labels
        np.save(f"{self.output_prefix}_full.npy", full)  # shape (N, 9)
        np.save(f"{self.output_prefix}_label.npy", labels)  # shape (N,)

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
    if data=="CW2":
        pcg = point_cloud_generator(rgb_file, depth_file, output_prefix_labels, fx, fy, cx, cy, do_inpaint=False)
        pcg.calculate(polygon_mask=mask) # or mask
    elif data=="Realsense":
        pcg = point_cloud_generator(rgb_file, depth_file, output_prefix_labels, fx, fy, cx, cy, do_inpaint=True)
        # pcg.depth = show_rgb_with_click_and_fit(rgb_file, pcg.depth, fx, fy, cx, cy)

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

        split_dir = os.path.join(base_dir, split)  # e.g. base_dir: data/CW2-Dataset/data, split: mit_32_d507
        if not os.path.isdir(split_dir):
            continue
        
        # os.listdir(split_dir): list all files and directories in the given path, e.g. ['mit_32_d507_1', 'mit_32_d507_2']
        # if split_dir has subfolders, process each subfolder
        subfolders = [os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for scene in subfolders:
            
            image_dir = os.path.join(scene, 'image')
            depth_dir = os.path.join(scene, 'depth') if os.path.exists(os.path.join(scene, 'depth')) else os.path.join(scene, 'depthTSDF') # depthTSDF is a subfolder of depth, while depth is a subfolder of scene
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
            print(f"length of image_files: {len(image_files)} in {image_dir}")
            print(f"length of depth_files: {len(depth_files)} in {depth_dir}")
            
            for polygon_list, img_file, depth_file in zip(tabletop_labels, image_files, depth_files):

                rgb_path = os.path.join(image_dir, img_file)    # RGB image path
                d_path = os.path.join(depth_dir, depth_file)    # depth image path
                filename_no_ext = os.path.splitext(img_file)[0] # remove the file extension

                # e.g. output_dir+f"/{dir_name}": data/dataset/point_clouds_C/train
                # e.g. os.path.basename(scene): mit_32_d507_1
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

def save_label_file_mapping(split_dir, output_file):
    """Scan split_dir and save (full.npy, label.npy) path pairs to a txt file."""
    entries = []
    for root, _, files in os.walk(split_dir):
        for file in files:
           
            if file.endswith("_full.npy"):
                full_path = os.path.join(root, file)
                label_path = full_path.replace('_full.npy', '_label.npy')
                if os.path.exists(label_path):  # Only record if label also exists
                    entries.append(f"{full_path} {label_path}\n")
      
    with open(output_file, "w") as f:
        f.writelines(entries)
    print(f"Saved {len(entries)} entries to {output_file}")


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


def process_all_single_folder(image_root, depth_root, intrinsic_path, output_dir):
 
    dir_name = 'test'

    if not os.path.exists(intrinsic_path):
        raise FileNotFoundError(f"intrinsics.txt not found in {intrinsic_path}")

    fx, fy, cx, cy = read_intrinsics_from_txt(intrinsic_path)
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    scenes = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]

    for scene in scenes:
        print(f"Processing scene: {scene}")
        # if scene != "big_yellow_square_table":
        #     continue
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
            generate_point_cloud(rgb_path, d_path, polygon_list, fx, fy, cx, cy, out_prefix, data="Realsense")


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
    data = "Realsense" # "Realsense"
    if data == "CW2":
        base_path = "data/CW2-Dataset/data"
        output_path = "data/dataset/point_clouds_C"
        process_all(base_path, output_path)
    
        save_label_file_mapping("data/dataset/point_clouds_C/train", "data/dataset/point_clouds_C/train/train_labels.txt")
        save_label_file_mapping("data/dataset/point_clouds_C/test", "data/dataset/point_clouds_C/test/test_labels.txt")
    elif data == "Realsense":
        # point cloud dataset organization
        base_path_realsense = "data/realsense"
        output_path_realsense = "data/dataset/realsense_point_clouds_C"

        image_root = os.path.join(base_path_realsense, 'image')
        depth_root = os.path.join(base_path_realsense, 'depth')
        intrinsic_path = os.path.join(base_path_realsense, 'intrinsics.txt')

        # generate npy & ply
        process_all_single_folder(image_root, depth_root, intrinsic_path, output_path_realsense)
        # genearte label files
        save_label_file_mapping("data/dataset/realsense_point_clouds_C/test", "data/dataset/realsense_point_clouds_C/test/test_labels.txt")
        scene_names = os.listdir(os.path.join(base_path_realsense+'/image'))
        print(scene_names)
        organize_rgbd_dataset_single_dataset(
            src_root=base_path_realsense,
            test_scenes=scene_names,
            dst_root='data/dataset/realsense_depth_img_C'
        )
        
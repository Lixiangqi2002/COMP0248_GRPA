import pickle
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image
import open3d as o3d
import cv2
import os


class point_cloud_generator():

    def __init__(self, rgb_file, depth_file, output_prefix, fx, fy, cx, cy, scalingfactor=1000):
        self.rgb_file = rgb_file
        self.depth_file = depth_file
        self.output_prefix = output_prefix  # e.g. "data/frame00001"
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scalingfactor = scalingfactor
        self.rgb = Image.open(rgb_file)
        self.depth = Image.open(depth_file).convert('I')  # uint16
        self.width = self.rgb.size[0]
        self.height = self.rgb.size[1]

    def calculate(self, polygon_mask=None):
        t1 = time.time()
        depth = np.asarray(self.depth).T  # (H, W)
        Z = depth / self.scalingfactor

        u = np.arange(self.width)
        v = np.arange(self.height)
        u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

        X = ((u_grid - self.cx) * Z) / self.fx
        Y = ((v_grid - self.cy) * Z) / self.fy

        img = np.array(self.rgb).transpose(1, 0, 2)  # (W, H, 3)

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
            polygon_mask = polygon_mask.T  # match (W, H)
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
            out_file = f"{self.output_prefix}_{tag}.ply"

            float_formatter = lambda x: "%.4f" % x
            points = []
            for i in df.T:
                points.append("{} {} {} {} {} {} 0\n".format(
                    float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                    int(i[3]), int(i[4]), int(i[5])
                ))

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

    def save_npy(self, label_array, suffix=''):
        os.makedirs(os.path.dirname(self.output_prefix), exist_ok=True)
        xyzrgb = self.df_full.T[:, :6]  # shape: (N, 6)
        data_with_label = np.concatenate([xyzrgb, label_array.reshape(-1, 1)], axis=1)  # shape: (N, 7)
        np.save(f"{self.output_prefix}{suffix}.npy", data_with_label)

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



def generate_point_cloud(rgb_file, depth_file, polygon_list, fx, fy, cx, cy, output_prefix_labels):
    if not polygon_list: 
        mask = np.zeros((480, 640), dtype=np.uint8)
    else:
        mask = create_polygon_mask(polygon_list, height=480, width=640)
    pcg = point_cloud_generator(rgb_file, depth_file, output_prefix_labels, fx, fy, cx, cy)
    pcg.calculate(polygon_mask=mask)
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
                generate_point_cloud(rgb_path, d_path, polygon_list, fx, fy, cx, cy, out_prefix)
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


def generate_label_file(split_dir, output_file):
    entries = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if file.endswith("_mask.npy"):
                # print(file)
                mask_path = os.path.join(root, file)
                pc_path = mask_path.replace('_mask.npy', '_full.npy')

                pc = np.load(pc_path)
                mask = np.load(mask_path)
                
                pc_rounded = np.round(pc, 3)
                mask_rounded = np.round(mask, 3)

                mask_set = set(map(tuple, mask_rounded))

                label = np.zeros(len(pc), dtype=np.int64)
                for i, p in enumerate(pc_rounded):
                    label[i] = 1 if tuple(p) in mask_set else 0

                # store labels in similar fashion to mask
                label_path = pc_path.replace("_full.npy", "_label.npy")
                np.save(label_path, label)
                entries.append(f"{pc_path} {label_path}\n")

    with open(output_file, "w") as f:
        f.writelines(entries)
    print(f"Saved {len(entries)} entries to {output_file}")



if __name__ == '__main__':
    import open3d as o3d

    pcd = o3d.io.read_point_cloud("data/dataset/point_clouds/train/mit_32_d507_d507_2_0000001-000000030183_full.npy")
    print("Has color:", pcd.has_colors())

    base_path = "data/CW2-Dataset/data"
    output_path = "data/dataset/point_clouds"
    process_all(base_path, output_path)
  
    generate_label_file("data/dataset/point_clouds/train", "data/dataset/point_clouds/train/train_labels.txt")
    generate_label_file("data/dataset/point_clouds/test", "data/dataset/point_clouds/test/test_labels.txt")
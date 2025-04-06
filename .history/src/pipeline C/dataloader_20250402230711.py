import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import open3d as o3d

# class PointCloudDataset(Dataset):
#     def __init__(self, label_file, num_points=1024, test = False):
#         """
#         label_file: txt file, each line like: /path/to/xxx_mask.npy 1
#         """
#         self.samples = []
#         self.num_points = num_points

#         with open(label_file, 'r') as f:
#             for line in f:
#                 path, label = line.strip().split()
#                 self.samples.append((path, int(label)))
#         self.test = test

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         pc = np.load(path)  # shape (N, 3)
#         # if self.test:
#         #     print(path, label)
#         #     ply_file  = path.replace('_full.npy', '_full.ply')
#         #     point_cloud = o3d.io.read_point_cloud(ply_file)
#         #     o3d.visualization.draw_geometries([point_cloud])

#         # Handle empty point clouds (should be rare)
#         if pc.shape[0] == 0:
#             pc = np.zeros((self.num_points, 3))
#         elif pc.shape[0] >= self.num_points:
#             indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
#             pc = pc[indices]
#         else:
#             indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
#             pc = pc[indices]

#         return torch.from_numpy(pc).float(), torch.tensor(label).long()

class PointCloudDataset(Dataset):
    def __init__(self, label_file, num_points=4096, test=False, augment=False):
        self.data = []
        self.labels = []
        self.num_points = num_points
        self.augment = augment
        self.test = test

        with open(label_file, "r") as f:
            for line in f:
                path, mask_path = line.strip().split()
                pc = np.load(path)  # shape: (N, 3)
                label = np.load(mask_path) # shape: (N,)
                self.data.append(pc)
                self.labels.append(label)

    def compute_weight(self):
        """Compute class weights based on the frequency of each class in the dataset."""
        # Suppose labels are integers from 0 to num_classes-1
        all_labels = np.concatenate(self.labels)
        class_counts = np.bincount(all_labels.astype(int))  # Compute frequency of each class
        total_count = len(all_labels)
        
        # Compute weights inversely proportional to class frequency
        # Avoid division by zero by adding a small constant (1e-6)
        class_counts = np.where(class_counts == 0, 1e-6, class_counts)
        self.weight = torch.tensor([total_count / class_counts[i] for i in range(len(class_counts))], dtype=torch.float32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc = self.data[idx]
        label = self.labels[idx]

        # print(f"Point cloud shape: {pc.shape}, Label shape: {label.shape}")
        assert pc.shape[0] == label.shape[0], "Point and label count mismatch!"


        if pc.shape[0] == 0:
            pc = np.zeros((self.num_points, 3))
            label = np.zeros((self.num_points,))
        elif pc.shape[0] >= self.num_points:  # downsample
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[indices]
            label = label[indices]
        else: # upsample
            indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
            pc = pc[indices]
            label = label[indices]

        # === Data Augmentation ===
        # no need to do this for labels
        if self.augment and not self.test:
            pc = self.random_rotate(pc)
            pc = self.random_jitter(pc)
            pc = self.random_scale(pc)

        return torch.tensor(pc, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def random_rotate(self, pc):
        """Random rotation around Z-axis"""
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        return pc @ rot_matrix.T

    def random_jitter(self, pc, sigma=0.01, clip=0.05):
        """Add random jittering (Gaussian noise)"""
        noise = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
        return pc + noise

    def random_scale(self, pc, scale_range=(0.8, 1.25)):
        """Random scaling"""
        scale = np.random.uniform(*scale_range)
        return pc * scale


if __name__=='__main__':

    trainset = PointCloudDataset("data/dataset/point_clouds/train/train_labels.txt", num_points=2048)
    testset = PointCloudDataset("data/dataset/point_clouds/test/test_labels.txt", num_points=2048) 
    train_loader = DataLoader(trainset, batch_size=16, shuffle=False)
    test_loader = DataLoader(testset, batch_size=16, shuffle=False)

    for i, (pc, label) in enumerate(train_loader):
        print(pc.shape, label.shape)
        if i == 0:
            break

    for i, (pc, label) in enumerate(test_loader):
        print(pc.shape, label.shape)
        if i == 0:
            break
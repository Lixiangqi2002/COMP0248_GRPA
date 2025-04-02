# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import open3d as o3d

# # class PointCloudDataset(Dataset):
# #     def __init__(self, label_file, num_points=1024, test = False):
# #         """
# #         label_file: txt file, each line like: /path/to/xxx_mask.npy 1
# #         """
# #         self.samples = []
# #         self.num_points = num_points

# #         with open(label_file, 'r') as f:
# #             for line in f:
# #                 path, label = line.strip().split()
# #                 self.samples.append((path, int(label)))
# #         self.test = test

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         path, label = self.samples[idx]
# #         pc = np.load(path)  # shape (N, 3)
# #         # if self.test:
# #         #     print(path, label)
# #         #     ply_file  = path.replace('_full.npy', '_full.ply')
# #         #     point_cloud = o3d.io.read_point_cloud(ply_file)
# #         #     o3d.visualization.draw_geometries([point_cloud])

# #         # Handle empty point clouds (should be rare)
# #         if pc.shape[0] == 0:
# #             pc = np.zeros((self.num_points, 3))
# #         elif pc.shape[0] >= self.num_points:
# #             indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
# #             pc = pc[indices]
# #         else:
# #             indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
# #             pc = pc[indices]

# #         return torch.from_numpy(pc).float(), torch.tensor(label).long()

# class PointCloudDataset(Dataset):
#     def __init__(self, label_file, num_points=4096, test=False, augment=False):
#         self.data = []
#         self.labels = []
#         self.num_points = num_points
#         self.augment = augment
#         self.test = test

#         with open(label_file, "r") as f:
#             for line in f:
#                 path, label = line.strip().split()
#                 pc = np.load(path)  # shape: (N, 3)
#                 self.data.append(pc)
#                 self.labels.append(int(label))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         pc = self.data[idx]
#         label = self.labels[idx]

#         if pc.shape[0] == 0:
#             pc = np.zeros((self.num_points, 3))
#         elif pc.shape[0] >= self.num_points:
#             indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
#             pc = pc[indices]
#         else:
#             indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
#             pc = pc[indices]

#         # === Data Augmentation ===
#         if self.augment and not self.test:
#             pc = self.random_rotate(pc)
#             pc = self.random_jitter(pc)
#             pc = self.random_scale(pc)

#         return torch.tensor(pc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#     def random_rotate(self, pc):
#         """Random rotation around Z-axis"""
#         theta = np.random.uniform(0, 2 * np.pi)
#         rot_matrix = np.array([
#             [np.cos(theta), -np.sin(theta), 0],
#             [np.sin(theta),  np.cos(theta), 0],
#             [0, 0, 1]
#         ])
#         return pc @ rot_matrix.T

#     def random_jitter(self, pc, sigma=0.01, clip=0.05):
#         """Add random jittering (Gaussian noise)"""
#         noise = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
#         return pc + noise

#     def random_scale(self, pc, scale_range=(0.8, 1.25)):
#         """Random scaling"""
#         scale = np.random.uniform(*scale_range)
#         return pc * scale


# if __name__=='__main__':
#     trainset = PointCloudDataset("data/dataset/point_clouds/train/train_labels.txt", num_points=2048)
#     testset = PointCloudDataset("data/dataset/point_clouds/test/test_labels.txt", num_points=2048) 
#     train_loader = DataLoader(trainset, batch_size=16, shuffle=False)
#     test_loader = DataLoader(testset, batch_size=16, shuffle=False)

#     for i, (pc, label) in enumerate(train_loader):
#         print(pc.shape, label.shape)
#         if i == 0:
#             break

#     for i, (pc, label) in enumerate(test_loader):
#         print(pc.shape, label.shape)
#         if i == 0:
#             break
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class TablePointCloudDataset(Dataset):
    def __init__(self, label_file, num_points=2048, use_fps=True, normalize=True, augment=False, test=False):
        self.paths = []
        self.labels = []
        self.num_points = num_points
        self.use_fps = use_fps
        self.normalize = normalize
        self.augment = augment
        self.test = test

        with open(label_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.paths.append(path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pc = np.load(self.paths[idx])  # (N, 3)
        label = self.labels[idx]

        # Sampling
        if pc.shape[0] == 0:
            pc = np.zeros((self.num_points, 3))
        elif pc.shape[0] >= self.num_points:
            if self.use_fps:
                pc = farthest_point_sample(pc, self.num_points)
            else:
                idxs = np.random.choice(pc.shape[0], self.num_points, replace=False)
                pc = pc[idxs]
        else:
            idxs = np.random.choice(pc.shape[0], self.num_points, replace=True)
            pc = pc[idxs]

        # Normalize
        if self.normalize:
            pc = pc_normalize(pc)

        # Augmentation
        if self.augment and not self.test:
            pc = self.random_rotate(pc)
            pc = self.random_jitter(pc)
            pc = self.random_scale(pc)

        return torch.tensor(pc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def random_rotate(self, pc):
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return pc @ rot_matrix.T

    def random_jitter(self, pc, sigma=0.01, clip=0.05):
        noise = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
        return pc + noise

    def random_scale(self, pc, scale_range=(0.8, 1.25)):
        scale = np.random.uniform(*scale_range)
        return pc * scale


if __name__ == '__main__':
    trainset = TablePointCloudDataset("data/dataset/point_clouds/train/train_labels.txt", num_points=2048, augment=True)
    testset = TablePointCloudDataset("data/dataset/point_clouds/test/test_labels.txt", num_points=2048, test=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    for pc, label in train_loader:
        print("Train batch:", pc.shape, label.shape)
        break

    for pc, label in test_loader:
        print("Test batch:", pc.shape, label.shape)
        break

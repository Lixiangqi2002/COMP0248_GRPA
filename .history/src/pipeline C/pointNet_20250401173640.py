import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataloader import PointCloudDataset
import os
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import open3d as o3d
import matplotlib.pyplot as plt
from DGCNN import DGCNN

class PointNetClassifier(nn.Module):
    def __init__(self):
        super(PointNetClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # binary classification 
        )

    def forward(self, x):
        x = x.transpose(2, 1)  # (B, 3, N)
        x = self.feature_extractor(x)  # (B, 256, N)
        x = torch.max(x, 2)[0]  # global feature (B, 256)
        x = self.classifier(x)  # (B, 2)
        return x



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

class PointNetSegmentation(nn.Module):
    def __init__(self):
        super(PointNetSegmentation, self).__init__()
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

        self.segmentation_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),  # (B, 256, N) -> (B, 128, N)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2, 1)     # output per-point classification (B, 2, N)
        )

    def forward(self, x):
        x = x.transpose(2, 1)  # (B, 3, N)
        features = self.feature_extractor(x)  # (B, 256, N)
        out = self.segmentation_head(features)  # (B, 2, N)
        return out.transpose(2, 1)  # (B, N, 2)


def train(train_loader, val_loader, device):
    # model = PointNetClassifier().to(device)
    model = DGCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_acc = 0.0

    train_losses = []
    val_losses = []

    for epoch in range(100):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for pc, label in train_loader:
            pc, label = pc.to(device), label.to(device).float()
            out = model(pc)

            loss = F.cross_entropy(out, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pc.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == label.long()).sum().item()
            total += label.size(0)

        acc = correct / total
        avg_train_loss = total_loss / total
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_total_loss = 0
        with torch.no_grad():
            for pc, label in val_loader:
                pc, label = pc.to(device), label.to(device).float()
                out = model(pc)
                loss = F.cross_entropy(out, label.long())
                val_total_loss += loss.item() * pc.size(0)
                pred = out.argmax(dim=1)
                val_correct += (pred == label.long()).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_total_loss / val_total
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Accuracy: {acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "weights/pipeline_A.pth")
            print("Saved best model to weights/pipeline_A.pth")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()


def test(test_loader, device):
    # model = PointNetClassifier().to(device)
    model = DGCNN().to(device)
    model.eval()

    model.load_state_dict(torch.load("weights/pipeline_A.pth", weights_only=True))
    print("Model loaded from weights/pipeline_A.pth")

    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for pc, label in test_loader:
            pc, label = pc.to(device), label.to(device).float()
            out = model(pc)
            loss = F.cross_entropy(out, label.long())

            total_loss += loss.item() * pc.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == label.long()).sum().item()
            total += label.size(0)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
           
            probs = torch.softmax(out, dim=1)
            confidences = probs.max(dim=1)[0]
            print("Avg Confidence:", confidences.mean().item())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Table", "Has Table"])
    disp.plot()
    plt.show()

    acc = correct / total
    avg_loss = total_loss / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = PointCloudDataset("data/dataset/point_clouds/train/train_labels.txt", num_points=4096, augment=True)
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    test_dataset = PointCloudDataset("data/dataset/point_clouds/test/test_labels.txt", num_points=4096, test=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    train(train_loader, val_loader, device)
    test(test_loader, device)

import os
import sys
import torch
import numpy as np
import datetime
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from dataloader import TablePointCloudDataset
from pointnet2_cls import Pointnet2, get_loss
from dataloader import TablePointCloudDataset

def parse_args():
    parser = argparse.ArgumentParser('PointNet2 Binary Classification in Pipeline A')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_point', type=int, default=2048, help='Point number per sample')
    parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_A', help='Log directory name')
    parser.add_argument('--resume', default=False,  action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--train_label_path', type=str, default='data/dataset/point_clouds/train/train_labels.txt', help='Path to training label file')
    return parser.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    experiment_dir = Path('./log/').joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(experiment_dir.joinpath(f"{args.log_dir}_train.txt"))
    logger.addHandler(file_handler)
    print(args)
    logger.info(args)

    # === Dataset split ===
    full_dataset = TablePointCloudDataset(args.train_label_path, num_points=args.num_point, augment=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # === Model ===
    classifier = Pointnet2(num_class=2, normal_channel=False)
    criterion = get_loss()
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        checkpoint_path = checkpoints_dir / 'best_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_acc = checkpoint.get('best_acc', 0.0)
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resumed from epoch {start_epoch}")
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            print("Checkpoint not found. Starting fresh.")
            logger.info("Checkpoint not found. Starting fresh.")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(start_epoch, args.epoch):
        print(f"Epoch {epoch+1}/{args.epoch}")
        logger.info(f"Epoch {epoch+1}/{args.epoch}")
        classifier.train()
        total_loss = 0
        total_correct = 0
        total_seen = 0

        for pc, label, _ in tqdm(train_loader):
 
            pc = pc.transpose(2, 1)
            
            if not args.use_cpu:
                pc, label = pc.cuda(), label.cuda()

            optimizer.zero_grad()
            pred, _ = classifier(pc)
            loss = criterion(pred, label, None)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pc.size(0)
            pred_choice = pred.max(1)[1]
            correct = pred_choice.eq(label).sum().item()
            total_correct += correct
            total_seen += label.size(0)

        train_acc = total_correct / total_seen
        train_loss = total_loss / total_seen
        print(f"Epoch {epoch+1}: Train Loss: {total_loss / total_seen:.4f} | Train Accuracy: {train_acc:.4f}")
        logger.info(f"Epoch {epoch+1}: Train Loss: {total_loss / total_seen:.4f} | Train Accuracy: {train_acc:.4f}")

        # === Validation ===
        total_seen = 0
        total_loss = 0
        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for pc, label , _  in val_loader:
                pc = pc.transpose(2, 1)
                if not args.use_cpu:
                    pc, label = pc.cuda(), label.cuda()
                pred, _ = classifier(pc)
                loss = criterion(pred, label, None)
                total_loss += loss.item() * pc.size(0)
                pred_choice = pred.max(1)[1]
                correct += pred_choice.eq(label).sum().item()
                total += label.size(0)
                total_seen += label.size(0)
            val_loss = total_loss / total_seen
            acc = correct / total
            print(f"Val Accuracy: {acc:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {acc:.4f}, Val Loss: {val_loss:.4f}")
            if acc > best_acc:
                best_acc = acc
                print("Saving best model...")
                logger.info("Saving best model...")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, checkpoints_dir.joinpath('best_model.pth'))

        scheduler.step()

        train_accuracies.append(train_acc)
        val_accuracies.append(acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print(f"Training completed. Best Val Accuracy: {best_acc:.4f}")
    logger.info(f"Training completed. Best Val Accuracy: {best_acc:.4f}")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses)+1))

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Val Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(experiment_dir.joinpath("loss_curve.png"))

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train/Val Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(experiment_dir.joinpath("accuracy_curve.png"))


if __name__ == '__main__':
    args = parse_args()
    main(args)

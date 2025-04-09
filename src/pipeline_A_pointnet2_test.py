import os
import sys
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from dataloader import TablePointCloudDataset
from pointnet2_cls import Pointnet2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def parse_args(data):
    if data == "CW2":
        parser = argparse.ArgumentParser('PointNet2 Binary Classification Testing in Pipeline A')
        parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_A', help='Experiment log directory')
        parser.add_argument('--test_label_path', type=str, default='data/dataset/point_clouds/test/test_labels.txt', help='Path to test label file')
    else:
        parser = argparse.ArgumentParser('PointNet2 Binary Classification Testing in Pipeline A RealSense')
        parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_A_realsense', help='Experiment log directory')
        parser.add_argument('--test_label_path', type=str, default='data/dataset/realsense_point_clouds/test_labels.txt', help='Path to test label file')

    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    return parser.parse_args()



def test(model, loader, logger, path):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, (pc, label, filename) in enumerate(tqdm(loader)):
            pc = pc.transpose(2, 1)
            pc, label = pc.cuda(), label.cuda()
            pred, _ = model(pc)
            pred_choice = pred.max(1)[1]

            y_true.extend(label.cpu().numpy())
            y_pred.extend(pred_choice.cpu().numpy())
            for f in filename:
                # logger.info(f"File: {f}, Predicted {pred_choice.cpu().numpy()==label.cpu().numpy()}")
                print(f"File: {f}, Predicted {pred_choice.cpu().numpy()==label.cpu().numpy()}")
            correct += pred_choice.eq(label).sum().item()
            total += label.size(0)

    accuracy = correct / total
    conf_mat = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=["No Table", "Table"], digits=4)
    print(f"\nConfusion Matrix:\n{conf_mat}")
    print(f"\nClassification Report:\n{class_report}")
    plot_confusion_matrix(y_true, y_pred, path)

    return accuracy, conf_mat, class_report



def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Table', 'Table'], yticklabels=['No Table', 'Table'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    experiment_dir = Path('./log/').joinpath(args.log_dir)
    checkpoint_path = 'log/binary_pointnet2_pipeline_A/checkpoints/best_model.pth'

    logger = logging.getLogger("Test")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(experiment_dir.joinpath(f"{args.log_dir}_test_metrics.txt"))
    logger.addHandler(file_handler)

    def log_string(msg):
        logger.info(msg)
        print(msg)

    log_string("Loading test dataset...")
    test_dataset = TablePointCloudDataset(args.test_label_path, num_points=args.num_point, test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    log_string("Loading model...")
    model = Pointnet2(num_class=2, normal_channel=False)
    if not args.use_cpu:
        model = model.cuda()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    log_string("Running evaluation...")
    accuracy, confusion_matrix, class_report = test(model, test_loader, logger, path=experiment_dir.joinpath('confusion_matrix.png'))
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix}")
    logger.info(f"\nClassification Report:\n{class_report}")
    log_string(f"Test Accuracy: {accuracy:.4f}")
 


if __name__ == '__main__':
    global args
    data = "Realsense" # "CW2" or "Realsense"
    args = parse_args(data)
    main(args)

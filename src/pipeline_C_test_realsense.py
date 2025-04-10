import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pointnet2_seg
from pipeline_C_dataloader import PointCloudDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['table', 'not_table']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=40960, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='binary_pointnet2_pipeline_C_realsense', required=False, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool



def visualize(pc, label):
    """Visualize point cloud with labels"""
    pc = pc.squeeze().cpu().numpy()  # shape: (9, N)
    xyz = pc[:3, :]  # keep only xyz coordinates
    xyz = xyz.transpose(1, 0)  # shape: (N, 3)
    
    label = label.squeeze().numpy()  # shape: (N,)

    # print(f"Shape of points: {xyz.shape}")
    # print(f"Shape of labels: {label.shape}")

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if pc.shape[0] >= 6:
        rgb = pc[3:6, :].T
        if np.max(rgb) > 1:
            rgb = rgb / 255.0
       
        rgb[label == 1] = [1.0, 0.0, 0.0]  
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        colors = np.zeros((xyz.shape[0], 3))
        colors[label == 0] = [0.5, 0.5, 0.5]   # background: gray
        colors[label == 1] = [1.0, 0.0, 0.0]   # table: red
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Visualize
    o3d.visualization.draw_geometries([pcd])


def compute_iou(pred, gt, class_id=1):
    """
    Compute IoU for a specific class.
    pred, gt: shape (N,)
    class_id: e.g., 1 for table
    """
    pred = pred.flatten()
    gt = gt.flatten()

    intersection = np.sum((pred == class_id) & (gt == class_id))
    union = np.sum((pred == class_id) | (gt == class_id))

    iou = intersection / union if union != 0 else 0.0
    return iou

def evaluate_binary_class(pred_labels, gt_labels, class_id=1, eps=1e-6):
  
    assert pred_labels.shape == gt_labels.shape

    pred_pos = (pred_labels == class_id)
    gt_pos = (gt_labels == class_id)

    TP = np.sum(pred_pos & gt_pos)
    FP = np.sum(pred_pos & (~gt_pos))
    FN = np.sum((~pred_pos) & gt_pos)

    iou = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return iou, precision, recall, f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    global_TP = 0
    global_FP = 0
    global_FN = 0

    NUM_CLASSES = 2  # 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    rotation_dict_path="data/dataset/realsense_point_clouds_C/rotation_matrices.json"
    if rotation_dict_path:
        with open(rotation_dict_path, "r") as f:
            rotation_dict = json.load(f)
    print("start loading testing data ...")
    test_dataset = PointCloudDataset(
        "data/dataset/realsense_point_clouds_C/test/test_labels.txt", 
        num_points=NUM_POINT, 
        test=True
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=True
        )

    '''MODEL LOADING'''
    model = pointnet2_seg.get_model(NUM_CLASSES).cuda()
    criterion = pointnet2_seg.get_loss().cuda()
    checkpoint = torch.load(str("log/binary_pointnet2_pipeline_C") + '/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    with torch.no_grad():
  
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        # for batch_idx in range(num_batches):
        for batch_idx, (points, labels, filename) in tqdm(enumerate(test_loader), total=len(test_loader)):
            print(filename)
           
            points = points.to(device)              # [B, N, 9]
            labels = labels.to(device).float()      # [B, N]
            
      
            points = points.transpose(2, 1)         # [B, N, 9] --> [B, 9, N]
            
            # initialize vote label pool
            vote_label_pool = np.zeros((labels.size(0), labels.size(1), NUM_CLASSES))  # [B, N, NUM_CLASSES]

            for _ in range(args.num_votes):
                seg_pred, _ = model(points)          # seg_pred: [B, N, NUM_CLASSES]
                # print("Distribution of predition labels:", seg_pred[0].cpu().data.numpy())
                pred_labels = torch.argmax(seg_pred, dim=-1).cpu().numpy()  # [B, N]
                unique, counts = np.unique(pred_labels, return_counts=True)
                print(dict(zip(unique, counts)))
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)  # seg_pred: [B*N, NUM_CLASSES]
                # pred_choice = seg_pred.cpu().data.max(1)[1]  # [B*N]
                pred_choice = torch.argmax(seg_pred, dim=1)  # [B*N]
                pred_choice = pred_choice.cpu()
                # reshape
                pred_choice_reshaped = pred_choice.reshape(points.size(0), points.size(2))  # [B, N]
                gt_labels = labels.cpu().numpy()  # [B, N]
                pred_labels = pred_choice_reshaped.numpy()  # [B, N]
                     
                # visualize(points, labels.cpu())
                visualize(points, pred_choice_reshaped)

                # iou_table = compute_iou(pred_labels, gt_labels)  # iou for table class
                # iou_background = compute_iou(pred_labels, gt_labels, class_id=0)  # iou for background class
                # print(f"Batch {batch_idx}, IoU of table: {iou_table:.4f}")
                # print(f"Batch {batch_idx}, IoU of background: {iou_background:.4f}")

                pred_np = pred_choice_reshaped.cpu().numpy() if hasattr(pred_choice, 'cpu') else pred_choice
                label_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels

                pred_bin = (pred_np == 1)
                label_bin = (label_np == 1)

                global_TP += np.logical_and(pred_bin, label_bin).sum()
                global_FP += np.logical_and(pred_bin, ~label_bin).sum()
                global_FN += np.logical_and(~pred_bin, label_bin).sum()

                # add votes to the vote label pool
                for b in range(points.size(0)):
                    for n in range(points.size(1)):
                        vote_label_pool[b, n, pred_choice[b * labels.size(1) + n]] += 1
                
            # get the final predicted label by taking the argmax over the vote label pool
            final_pred = np.argmax(vote_label_pool, axis=2)  # pred_label: [B, N]
            
            # calculate the loss
            loss = criterion(seg_pred, labels.view(-1).long(), trans_feat=None, weight=None)
            loss_sum += loss.item() * points.size(0)

            # correct = (final_pred == labels.cpu().numpy()).sum().item()

            total_seen_class_tmp = np.zeros(NUM_CLASSES)
            total_correct_class_tmp = np.zeros(NUM_CLASSES)
            total_iou_deno_class_tmp = np.zeros(NUM_CLASSES)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((labels.cpu().numpy() == l))
                total_correct_class_tmp[l] += np.sum((final_pred == l) & (labels.cpu().numpy() == l))
                total_iou_deno_class_tmp[l] += np.sum(((final_pred == l) | (labels.cpu().numpy() == l)))

            for l in range(NUM_CLASSES):
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

        print("Evaluation complete!")

if __name__ == '__main__':
    args = parse_args()
    main(args)

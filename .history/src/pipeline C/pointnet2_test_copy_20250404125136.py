"""
Author: Benny
Date: Nov 2019
"""
import argparse
# 将项目根目录添加到 sys.path 中
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models import pointnet2_seg
from dataloader import PointCloudDataset
from torch.utils.data import DataLoader, random_split

import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

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
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='2025-04-03_17-05', required=False, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

import open3d as o3d
import numpy as np


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

    # Set colors based on labels
    colors = np.zeros((xyz.shape[0], 3))
    colors[label == 0] = [0.5, 0.5, 0.5]   # gray
    colors[label == 1] = [1.0, 0.0, 0.0]   # red
    pcd.colors = o3d.utility.Vector3dVector(colors)

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
    """
    评估单一类别（例如 table）的分割性能
    pred_labels: (N,) numpy array, 预测结果
    gt_labels:   (N,) numpy array, 真实标签
    class_id:    int, 要评估的类别编号，默认是 1 (table)
    """
    assert pred_labels.shape == gt_labels.shape

    # 只关注当前 class 的 TP, FP, FN
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
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

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

    NUM_CLASSES = 2  # 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    
    print("start loading testing data ...")
    test_dataset = PointCloudDataset(
        "data/dataset/point_clouds/test/test_labels.txt", 
        num_points=NUM_POINT, 
        test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        drop_last=True
        )
    # root = 'data/s3dis/stanford_indoor3d/'

    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    # log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = pointnet2_seg.get_model(NUM_CLASSES).cuda()
    criterion = pointnet2_seg.get_loss().cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()

    with torch.no_grad():
        # scene_id = test_dataset.file_list
        # scene_id = [x[:-4] for x in scene_id]
        # num_batches = len(test_dataset)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        # for batch_idx in range(num_batches):
        for batch_idx, (points, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            points = points.to(device)              # [B, N, 9]
            labels = labels.to(device).float()      # [B, N]

            # print(f"Shape of points: {points.shape}")
            # print(f"Shape of labels: {labels.shape}")
            points = points.transpose(2, 1)         # [B, N, 9] --> [B, 9, N]
            
            # initialize vote label pool
            vote_label_pool = np.zeros((labels.size(0), labels.size(1), NUM_CLASSES))  # [B, N, NUM_CLASSES]

            for _ in range(args.num_votes):
                seg_pred, _ = model(points)          # seg_pred: [B, N, NUM_CLASSES]
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)  # seg_pred: [B*N, NUM_CLASSES]
                pred_choice = seg_pred.cpu().data.max(1)[1]  # [B*N]

                # reshape
                pred_choice_reshaped = pred_choice.reshape(points.size(0), points.size(2))  # [B, N]
                gt_labels = labels.cpu().numpy()  # [B, N]
                pred_labels = pred_choice_reshaped.numpy()  # [B, N]
                
                # visualize(points, labels.cpu())
                # visualize(points, pred_choice_reshaped)

                # iou_table = compute_iou(pred_labels, gt_labels)  # iou for table class
                # iou_background = compute_iou(pred_labels, gt_labels, class_id=0)  # iou for background class
                # print(f"Batch {batch_idx}, IoU of table: {iou_table:.4f}")
                # print(f"Batch {batch_idx}, IoU of background: {iou_background:.4f}")

                pred_np = pred_choice_reshaped.cpu().numpy() if hasattr(pred_choice, 'cpu') else pred_choice
                label_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
                iou, p, r, f1 = evaluate_binary_class(pred_np, label_np, class_id=1)
                print(f"Table: IoU={iou:.4f}, Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")


                # add votes to the vote label pool
                for b in range(points.size(0)):
                    for n in range(points.size(1)):
                        vote_label_pool[b, n, pred_choice[b * labels.size(1) + n]] += 1
                
            # get the final predicted label by taking the argmax over the vote label pool
            final_pred = np.argmax(vote_label_pool, axis=2)  # pred_label: [B, N]

            # calculate the loss
            loss = criterion(seg_pred, labels.view(-1).long(), trans_feat=None, weight=None)
        
            correct = (final_pred == labels.cpu().numpy()).sum().item()

            total_seen_class_tmp = np.zeros(NUM_CLASSES)
            total_correct_class_tmp = np.zeros(NUM_CLASSES)
            total_iou_deno_class_tmp = np.zeros(NUM_CLASSES)

            # 计算每个类的 IoU 和正确数
            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((labels.cpu().numpy() == l))
                total_correct_class_tmp[l] += np.sum((final_pred == l) & (labels.cpu().numpy() == l))
                total_iou_deno_class_tmp[l] += np.sum(((final_pred == l) | (labels.cpu().numpy() == l)))

            # 将每个批次的统计结果累加
            for l in range(NUM_CLASSES):
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            # 计算每个批次的 IoU
            iou_map = total_correct_class_tmp / (total_iou_deno_class_tmp + 1e-6)
            tmp_iou = np.mean(iou_map[total_seen_class_tmp != 0])
            # log_string(f'Mean IoU of batch {batch_idx}: {tmp_iou:.4f}')
            # print(f'Batch {batch_idx}, IoU: {tmp_iou:.4f}')

        # 计算总体的 IoU 和准确率
        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
        log_string('------- IoU --------')
        for l in range(NUM_CLASSES):
            log_string(f'class {seg_label_to_cat[l]}, IoU: {IoU[l]:.3f}')

        # 最终评估结果
        log_string(f'Final eval point avg class IoU: {np.mean(IoU):.4f}')
        log_string(f'Final eval point accuracy: {np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6):.4f}')

        print("Evaluation complete!")

        #     correct = (pred == labels.long()).sum().item()
        #     print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
        #     total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
        #     total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
        #     total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
        #     if args.visual:
        #         fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
        #         fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

        #     whole_scene_data = test_dataset.scene_points_list[batch_idx]
        #     whole_scene_label = test_dataset.semantic_labels_list[batch_idx]
        #     vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        #     for _ in tqdm(range(args.num_votes), total=args.num_votes):
        #         scene_data, scene_label, scene_smpw, scene_point_index = test_dataset[batch_idx]
        #         num_blocks = scene_data.shape[0]
        #         s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
        #         batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

        #         batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
        #         batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
        #         batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

        #         for sbatch in range(s_batch_num):
        #             start_idx = sbatch * BATCH_SIZE
        #             end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
        #             real_batch_size = end_idx - start_idx
        #             batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
        #             batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
        #             batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
        #             batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
        #             batch_data[:, :, 3:6] /= 1.0

        #             torch_data = torch.Tensor(batch_data)
        #             torch_data = torch_data.float().cuda()
        #             torch_data = torch_data.transpose(2, 1)
        #             seg_pred, _ = model(torch_data)
        #             batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

        #             vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
        #                                        batch_pred_label[0:real_batch_size, ...],
        #                                        batch_smpw[0:real_batch_size, ...])

        #     pred_label = np.argmax(vote_label_pool, 1)

        #     for l in range(NUM_CLASSES):
        #         total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
        #         total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
        #         total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
        #         total_seen_class[l] += total_seen_class_tmp[l]
        #         total_correct_class[l] += total_correct_class_tmp[l]
        #         total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

        #     iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
        #     print(iou_map)
        #     arr = np.array(total_seen_class_tmp)
        #     tmp_iou = np.mean(iou_map[arr != 0])
        #     log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
        #     print('----------------------------')

        #     filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
        #     with open(filename, 'w') as pl_save:
        #         for i in pred_label:
        #             pl_save.write(str(int(i)) + '\n')
        #         pl_save.close()
        #     for i in range(whole_scene_label.shape[0]):
        #         color = g_label2color[pred_label[i]]
        #         color_gt = g_label2color[whole_scene_label[i]]
        #         if args.visual:
        #             fout.write('v %f %f %f %d %d %d\n' % (
        #                 whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
        #                 color[2]))
        #             fout_gt.write(
        #                 'v %f %f %f %d %d %d\n' % (
        #                     whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
        #                     color_gt[1], color_gt[2]))
        #     if args.visual:
        #         fout.close()
        #         fout_gt.close()

        # IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        # iou_per_class_str = '------- IoU --------\n'
        # for l in range(NUM_CLASSES):
        #     iou_per_class_str += 'class %s, IoU: %.3f \n' % (
        #         seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
        #         total_correct_class[l] / float(total_iou_deno_class[l]))
        # log_string(iou_per_class_str)
        # log_string('eval point avg class IoU: %f' % np.mean(IoU))
        # log_string('eval whole scene point avg class acc: %f' % (
        #     np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        # log_string('eval whole scene point accuracy: %f' % (
        #         np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
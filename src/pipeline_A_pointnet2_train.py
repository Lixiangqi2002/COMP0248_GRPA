# """
# Author: Benny
# Date: Nov 2019
# """

# import os
# import sys
# import torch
# import numpy as np

# import datetime
# import logging
# import provider
# import importlib
# import shutil
# import argparse

# from pathlib import Path
# from tqdm import tqdm
# from data_utils.ModelNetDataLoader import ModelNetDataLoader

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

# def parse_args():
#     '''PARAMETERS'''
#     parser = argparse.ArgumentParser('training')
#     parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
#     parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
#     parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
#     parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
#     parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
#     parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
#     parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
#     parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
#     parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
#     parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
#     parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
#     parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
#     parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
#     parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
#     return parser.parse_args()


# def inplace_relu(m):
#     classname = m.__class__.__name__
#     if classname.find('ReLU') != -1:
#         m.inplace=True


# def test(model, loader, num_class=40):
#     mean_correct = []
#     class_acc = np.zeros((num_class, 3))
#     classifier = model.eval()

#     for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

#         if not args.use_cpu:
#             points, target = points.cuda(), target.cuda()

#         points = points.transpose(2, 1)
#         pred, _ = classifier(points)
#         pred_choice = pred.data.max(1)[1]

#         for cat in np.unique(target.cpu()):
#             classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
#             class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
#             class_acc[cat, 1] += 1

#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item() / float(points.size()[0]))

#     class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
#     class_acc = np.mean(class_acc[:, 2])
#     instance_acc = np.mean(mean_correct)

#     return instance_acc, class_acc


# def main(args):
#     def log_string(str):
#         logger.info(str)
#         print(str)

#     '''HYPER PARAMETER'''
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#     '''CREATE DIR'''
#     timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
#     exp_dir = Path('./log/')
#     exp_dir.mkdir(exist_ok=True)
#     exp_dir = exp_dir.joinpath('classification')
#     exp_dir.mkdir(exist_ok=True)
#     if args.log_dir is None:
#         exp_dir = exp_dir.joinpath(timestr)
#     else:
#         exp_dir = exp_dir.joinpath(args.log_dir)
#     exp_dir.mkdir(exist_ok=True)
#     checkpoints_dir = exp_dir.joinpath('checkpoints/')
#     checkpoints_dir.mkdir(exist_ok=True)
#     log_dir = exp_dir.joinpath('logs/')
#     log_dir.mkdir(exist_ok=True)

#     '''LOG'''
#     args = parse_args()
#     logger = logging.getLogger("Model")
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#     log_string('PARAMETER ...')
#     log_string(args)

#     '''DATA LOADING'''
#     log_string('Load dataset ...')
#     data_path = 'data/modelnet40_normal_resampled/'

#     train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
#     test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
#     trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
#     testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

#     '''MODEL LOADING'''
#     num_class = args.num_category
#     model = importlib.import_module(args.model)
#     shutil.copy('./models/%s.py' % args.model, str(exp_dir))
#     shutil.copy('models/pointnet2_utils.py', str(exp_dir))
#     shutil.copy('./train_classification.py', str(exp_dir))

#     classifier = model.get_model(num_class, normal_channel=args.use_normals)
#     criterion = model.get_loss()
#     classifier.apply(inplace_relu)

#     if not args.use_cpu:
#         classifier = classifier.cuda()
#         criterion = criterion.cuda()

#     try:
#         checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
#         start_epoch = checkpoint['epoch']
#         classifier.load_state_dict(checkpoint['model_state_dict'])
#         log_string('Use pretrain model')
#     except:
#         log_string('No existing model, starting training from scratch...')
#         start_epoch = 0

#     if args.optimizer == 'Adam':
#         optimizer = torch.optim.Adam(
#             classifier.parameters(),
#             lr=args.learning_rate,
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             weight_decay=args.decay_rate
#         )
#     else:
#         optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
#     global_epoch = 0
#     global_step = 0
#     best_instance_acc = 0.0
#     best_class_acc = 0.0

#     '''TRANING'''
#     logger.info('Start training...')
#     for epoch in range(start_epoch, args.epoch):
#         log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
#         mean_correct = []
#         classifier = classifier.train()

#         scheduler.step()
#         for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
#             optimizer.zero_grad()

#             points = points.data.numpy()
#             points = provider.random_point_dropout(points)
#             points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
#             points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
#             points = torch.Tensor(points)
#             points = points.transpose(2, 1)

#             if not args.use_cpu:
#                 points, target = points.cuda(), target.cuda()

#             pred, trans_feat = classifier(points)
#             loss = criterion(pred, target.long(), trans_feat)
#             pred_choice = pred.data.max(1)[1]

#             correct = pred_choice.eq(target.long().data).cpu().sum()
#             mean_correct.append(correct.item() / float(points.size()[0]))
#             loss.backward()
#             optimizer.step()
#             global_step += 1

#         train_instance_acc = np.mean(mean_correct)
#         log_string('Train Instance Accuracy: %f' % train_instance_acc)

#         with torch.no_grad():
#             instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

#             if (instance_acc >= best_instance_acc):
#                 best_instance_acc = instance_acc
#                 best_epoch = epoch + 1

#             if (class_acc >= best_class_acc):
#                 best_class_acc = class_acc
#             log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
#             log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

#             if (instance_acc >= best_instance_acc):
#                 logger.info('Save model...')
#                 savepath = str(checkpoints_dir) + '/best_model.pth'
#                 log_string('Saving at %s' % savepath)
#                 state = {
#                     'epoch': best_epoch,
#                     'instance_acc': instance_acc,
#                     'class_acc': class_acc,
#                     'model_state_dict': classifier.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                 }
#                 torch.save(state, savepath)
#             global_epoch += 1

#     logger.info('End of training...')


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

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
import provider

def parse_args():
    parser = argparse.ArgumentParser('PointNet2 Binary Classification')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_point', type=int, default=2048, help='Point number per sample')
    parser.add_argument('--log_dir', type=str, default='binary_pointnet2', help='Log directory name')
    parser.add_argument('--resume', default=True,  action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--train_label_path', type=str, default='data/dataset/point_clouds/train/train_labels.txt', help='Path to training label file')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    experiment_dir = Path('./log/').joinpath(args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(experiment_dir.joinpath(f"{args.log_dir}_train_A.txt"))
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

        for pc, label in tqdm(train_loader):
            # pc = pc.transpose(2, 1)
            pc = pc.data.numpy()
            pc = provider.random_point_dropout(pc)
            pc[:, :, 0:3] = provider.random_scale_point_cloud(pc[:, :, 0:3])
            pc[:, :, 0:3] = provider.shift_point_cloud(pc[:, :, 0:3])
            pc = torch.Tensor(pc)
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
        print(f"Epoch {epoch+1}: Train Loss: {total_loss / total_seen:.4f} | Train Accuracy: {train_acc:.4f}")
        logger.info(f"Epoch {epoch+1}: Train Loss: {total_loss / total_seen:.4f} | Train Accuracy: {train_acc:.4f}")

        # === Validation ===
        classifier.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for pc, label in val_loader:
                pc = pc.transpose(2, 1)
                if not args.use_cpu:
                    pc, label = pc.cuda(), label.cuda()
                pred, _ = classifier(pc)
                pred_choice = pred.max(1)[1]
                correct += pred_choice.eq(label).sum().item()
                total += label.size(0)

            acc = correct / total
            print(f"Val Accuracy: {acc:.4f}")
            logger.info(f"Val Accuracy: {acc:.4f}")
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
        train_losses.append(total_loss / total_seen)
        val_losses.append(acc)

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
    main()

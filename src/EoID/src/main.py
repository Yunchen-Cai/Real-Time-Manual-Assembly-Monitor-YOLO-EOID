import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import json


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float)  # 主学习率减小
    parser.add_argument('--lr_backbone', default=5e-6, type=float)  # Backbone 学习率减小
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--save_all', default=True, type=bool)


    # zero-shot with CLIP
    parser.add_argument('--model', default='eoid', type=str,
                        choices=['cdn', 'eoid', 'eoid_acc', 'cons', 'eoid_gen_acc'])
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--thres', default=0.5, type=float)
    parser.add_argument('--inter_score', action='store_true')
    parser.add_argument('--topk_is', action='store_true')
    parser.add_argument('--gtclip', action='store_true')
    parser.add_argument('--neg_0', action='store_true')
    parser.add_argument('--vdetach', action='store_true')
    parser.add_argument('--verb_loss_type', default='focal_bce', type=str, choices=['bce_bce', 'focal_bce', ])
    parser.add_argument('--clip_backbone', default='RN50', choices=['RN50', 'RN50x16', 'RN101', 'ViT-B-32', 'ViT-B-16'])
    parser.add_argument('--uc_type', default='uc0', type=str,
                        choices=['rare_first', 'non_rare_first', 'uc0', 'uc1', 'uc2', 'uc3', 'uc4'],
                        help='Select uc_type, uc0~4 denote default five uc types')
    parser.add_argument('--clipseen_reweight', default=False, type=bool)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="Number of interaction decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_matching', default=1, type=float,
                        help="Sub and obj box matching coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=3, type=float)
    parser.add_argument('--giou_loss_coef', default=3.0, type=float)
    parser.add_argument('--obj_loss_coef', default=3, type=float)
    parser.add_argument('--verb_loss_coef', default=3, type=float)
    parser.add_argument('--clip_loss_coef', default=2, type=float)
    parser.add_argument('--distill_loss_coef', default=2, type=float)
    parser.add_argument('--is_loss_coef', default=1, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='rice_cooker')

    # Assembly101
    parser.add_argument('--assembly101_path', default='J:\\dataset_ass101', type=str,
                        help='path to the Assembly101 dataset')
    parser.add_argument('--num_verbs', type=int, default=200, help='number of verb classes')  # 确保范围足够
    parser.add_argument('--num_objects', type=int, default=53, help='number of object classes')
    # parser.add_argument('--num_verbs', type=int, default=10, help='number of verb classes')

    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:29500', help='url used to set up distributed training')

    # decoupling training parameters
    parser.add_argument('--freeze_mode', default=0, type=int)
    parser.add_argument('--obj_reweight', action='store_true')
    parser.add_argument('--verb_reweight', action='store_true')
    parser.add_argument('--use_static_weights', action='store_true',
                        help='use static weights or dynamic weights, default use dynamic')
    parser.add_argument('--queue_size', default=4704 * 1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_verb', default=0.7, type=float,
                        help='Reweighting parameter for verb')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    # 模型迁移
    parser.add_argument('--use_bbox', default=False)
    parser.add_argument('--use_clip', default=False)

    # Dataset parameters
    parser.add_argument('--rice_cooker_path', type=str, required=True)  # 图像文件夹路径
    # parser.add_argument('--img_folder', type=str, required=True, help='Path to the image folder')  # 图像文件夹路径
    # parser.add_argument('--anno_file', type=str, required=True, help='Path to the annotation file')  # 标注文件路径

    return parser

def plot_performance_trends(performance_trends):
    epochs = performance_trends['epoch']

    # 绘制 mAP 趋势
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, performance_trends['mAP'], label="mAP", marker='o')
    plt.title("mAP Performance Trends")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_mAP_trend.png")
    plt.close()

    # 绘制 Verb Match Precision/Recall 趋势
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, performance_trends['verb_match_precision'], label="Verb Match Precision", marker='o')
    plt.plot(epochs, performance_trends['verb_match_recall'], label="Verb Match Recall", marker='s')
    plt.title("Verb Match Performance Trends")
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_verb_match_trend.png")
    plt.close()

    # 绘制 IoU Match Precision/Recall 趋势
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, performance_trends['iou_match_precision'], label="IoU Match Precision", marker='o')
    plt.plot(epochs, performance_trends['iou_match_recall'], label="IoU Match Recall", marker='s')
    plt.title("IoU Match Performance Trends")
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_iou_match_trend.png")
    plt.close()



def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)
    # 初始化性能记录数据结构
    performance_trends = {
        'epoch': [],
        'mAP': [],
        'verb_match_precision': [],
        'verb_match_recall': [],
        'iou_match_precision': [],
        'iou_match_recall': []
    }

    # 数据集的选择和参数
    if args.dataset_file == 'rice_cooker':
        num_obj_classes = args.num_obj_classes
        num_verb_classes = args.num_verb_classes
    else:
        raise ValueError(f"Dataset {args.dataset_file} not supported")

    model, criterion, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume,map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    # Training setup
    print("Start training")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # DataLoader
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # Optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    for images, targets in data_loader_train:
        print(targets)
        break
    # 初始化记录列表
    loss_obj_ce_history = []
    loss_verb_ce_history = []
    loss_sub_bbox_history = []
    loss_obj_bbox_history = []
    loss_sub_giou_history = []
    loss_obj_giou_history = []
    # Start training
    start_time = time.time()
    best_performance = -1
    from collections import Counter
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}")

        # 调用 set_epoch 确保分布式采样一致
        sampler_train.set_epoch(epoch)

        # 打乱数据集标注文件
        random.shuffle(dataset_train.annotations)

        # 打印 action_id 的分布
        action_ids = [item['annotations']['action_id'] for item in dataset_train.annotations]
        print(f"Epoch {epoch + 1}, Action ID Distribution: {Counter(action_ids)}")

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        # 保存损失值
        loss_obj_ce_history.append(train_stats["loss_obj_ce"])
        loss_verb_ce_history.append(train_stats["loss_verb_ce"])
        loss_sub_bbox_history.append(train_stats.get('loss_sub_bbox', 0))  # 如果不存在则用 0
        loss_obj_bbox_history.append(train_stats.get('loss_obj_bbox', 0))
        loss_sub_giou_history.append(train_stats.get('loss_sub_giou', 0))
        loss_obj_giou_history.append(train_stats.get('loss_obj_giou', 0))
        print(f"Epoch {epoch + 1}:")
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
        performance = test_stats['mAP']
        verb_match_ratio = test_stats.get('verb_match_ratio', 0)
        iou_match_ratio = test_stats.get('iou_match_ratio', 0)
        print(f"Verb Match Ratio: {verb_match_ratio:.4f}")
        print(f"IoU Match Ratio: {iou_match_ratio:.4f}")
        # 保存当前 epoch 的性能数据
        performance_trends['epoch'].append(epoch + 1)
        performance_trends['mAP'].append(performance)
        performance_trends['verb_match_precision'].append(test_stats.get('verb_match_precision', 0.0))
        performance_trends['verb_match_recall'].append(test_stats.get('verb_match_recall', 0.0))
        performance_trends['iou_match_precision'].append(test_stats.get('iou_match_precision', 0.0))
        performance_trends['iou_match_recall'].append(test_stats.get('iou_match_recall', 0.0))



        print(f"Calling evaluate_hoi with args: {args}")
        # 跳过 evaluate_hoi，假装 mAP 为 0.0
        # performance = 0.0
        # print(f"Epoch {epoch} Performance: {performance}")
        output_dir = args.output_dir

        if performance > best_performance:
            best_performance = performance
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')

            if not os.path.exists(output_dir):
                print(f"Creating directory: {output_dir}")
                os.makedirs(output_dir)
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

        if args.freeze_mode == 0 and epoch < args.lr_drop and epoch % 5 != 0:  # eval every 5 epoch before lr_drop
            continue
        elif args.freeze_mode == 0 and epoch >= args.lr_drop and epoch % 2 == 0:  # eval every 2 epoch after lr_drop
            continue

        if args.save_all:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{epoch}.pth')
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'criterion': criterion.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        print(f"Epoch {epoch} Performance: {performance}")
    # 绘制趋势图
    plot_performance_trends(performance_trends)
    # 在训练完成后保存损失值
    with open("loss_history.json", "w") as f:
        json.dump({
            "loss_obj_ce": loss_obj_ce_history,
            "loss_verb_ce": loss_verb_ce_history
        }, f)
    print(f"Loss history saved successfully to {os.path.abspath('loss_history.json')}")

    # 绘制损失曲线
    epochs = list(range(1, len(loss_obj_ce_history) + 1))  # 确保长度一致
    plt.plot(epochs, loss_obj_ce_history, label="loss_obj_ce")
    plt.plot(epochs, loss_verb_ce_history, label="loss_verb_ce")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.savefig("loss_curve.png")  # 保存曲线到文件
    # plt.show()
    print(f"Loss curve saved successfully to {os.path.abspath('loss_curve.png')}")
    # 每个 epoch 后保存趋势图
    # 绘制 bbox 和 giou 损失趋势图
    plt.figure()
    plt.plot(loss_sub_bbox_history, label="Sub BBox Loss")
    plt.plot(loss_obj_bbox_history, label="Obj BBox Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("BBox Loss (Sub and Obj) Over Epochs")
    plt.savefig("loss_bbox_trend.png")
    plt.close()

    plt.figure()
    plt.plot(loss_sub_giou_history, label="Sub GIoU Loss")
    plt.plot(loss_obj_giou_history, label="Obj GIoU Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GIoU Loss (Sub and Obj) Over Epochs")
    plt.savefig("loss_giou_trend.png")
    plt.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed in {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

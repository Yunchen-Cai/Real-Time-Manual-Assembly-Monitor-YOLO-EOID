import math
import os
import sys
from typing import Iterable
from xmlrpc.client import boolean
import numpy as np
import copy
import itertools
import time

import torch

import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.hico_ua_eval import HICOUAEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from datasets.vcoco_pseudo import VCOCOPse
from datasets.hvcoco_eval import HVCOEvaluator
from datasets.ricecooker_eval import RiceCookerEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    start_time = time.time()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        for t in targets:
            for k, v in t.items():
                if k not in ['filename', 'img_or']:
                    t[k] = t[k].to(device)

        data_time = time.time()
        outputs = model(samples)
        forward_time = time.time()
        loss_dict = criterion(outputs, targets)
        loss_time = time.time()
        weight_dict = criterion.weight_dict

        # 提取预测的 verb_scores 和对应的预测 ID
        pred_verb_scores = outputs['pred_verb_logits']
        pred_verb_ids = torch.argmax(pred_verb_scores, dim=2)  # 假设 verb_logits 是 [batch_size, num_queries, num_classes]
        #
        # 打印预测和 Ground Truth 的 Verb ID
        for pred_ids, target in zip(pred_verb_ids, targets):
            gt_verb_ids = [torch.argmax(v_label).item() for v_label in target['verb_labels']]
            # print(f"Predicted Verb IDs: {pred_ids.tolist()}")
            # print(f"Ground Truth Verb IDs: {gt_verb_ids}")
        # # 打印当前 batch 的损失值
        # print(f"Loss Dict: {loss_dict}")

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        # 检查 bbox_embed 的梯度
        if hasattr(model, 'bbox_embed') and hasattr(model.bbox_embed[-1], 'weight'):
            print(f"Gradient of bbox_embed: {model.bbox_embed[-1].weight.grad}")

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # print('data_time:', data_time - start_time)
        # print('forward_time:', forward_time - data_time)
        # print('loss_time:', loss_time - forward_time)
        start_time = time.time()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 1, header):
        samples = samples.to(device)

        outputs = model(samples)

        # 提取 verb_scores 并计算 pred_verb_ids
        from collections import Counter

        # 获取当前 batch 的 GT Verb ID
        gt_verb_ids = [torch.argmax(t["verb_labels"]).item() for t in targets]  # 提取 GT Verb ID
        gt_distribution = Counter(gt_verb_ids)  # 统计当前 batch 里的 Verb ID 分布

        # 打印 GT Verb ID 的分布情况
        print(f"🔍 GT Verb ID Distribution in Current Batch: {gt_distribution}")

        # 继续处理 logits 和 predicted verb ID
        verb_scores = outputs['pred_verb_logits'].sigmoid()
        # print(f"🔍 Raw Verb Scores: {verb_scores[0, :10]}")  # 只打印前 10 个样本的分数

        pred_verb_ids = torch.argmax(outputs['pred_verb_logits'], dim=2)  # ✅ 直接用 logits 计算 argmax
        print(f"🔍 Predicted Verb IDs (after argmax): {pred_verb_ids.tolist()}")

        # 打印分布信息
        unique_ids, counts = torch.unique(pred_verb_ids, return_counts=True)
        print(f"Predicted Verb ID Distribution: {dict(zip(unique_ids.tolist(), counts.tolist()))}")

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # filenames = [t['filename'] for t in targets]
        import matplotlib.pyplot as plt
        import cv2
        results = postprocessors['hoi'](outputs, orig_target_sizes)
        # for t, r in zip(targets, results):
        #     # 转换原始图像为 NumPy 格式
        #     img = np.array(t['img_or'])
        #
        #     # 获取 Ground Truth 和预测框
        #     gt_obj_box = t['obj_boxes'][0].cpu().numpy()
        #     gt_sub_box = t['sub_boxes'][0].cpu().numpy()
        #
        #     pred_obj_box = r['boxes'][0].numpy()  # 预测的 obj_box
        #     pred_sub_box = r['boxes'][1].numpy()  # 预测的 sub_box
        #
        #     # 检查边界框坐标
        #     print("Ground Truth Object Box:", gt_obj_box)
        #     print("Ground Truth Subject Box:", gt_sub_box)
        #     print("Predicted Object Box:", pred_obj_box)
        #     print("Predicted Subject Box:", pred_sub_box)
        #
        #     # 绘制 Ground Truth 的 object 和 subject 框
        #     img = cv2.rectangle(img, (int(gt_obj_box[0]), int(gt_obj_box[1])),
        #                         (int(gt_obj_box[2]), int(gt_obj_box[3])), (0, 255, 0), 2)  # Green: Ground Truth Object
        #     img = cv2.rectangle(img, (int(gt_sub_box[0]), int(gt_sub_box[1])),
        #                         (int(gt_sub_box[2]), int(gt_sub_box[3])), (255, 255, 0),
        #                         2)  # Yellow: Ground Truth Subject
        #
        #     # 绘制预测的 object 和 subject 框
        #     img = cv2.rectangle(img, (int(pred_obj_box[0]), int(pred_obj_box[1])),
        #                         (int(pred_obj_box[2]), int(pred_obj_box[3])), (255, 0, 0), 2)  # Red: Predicted Object
        #     img = cv2.rectangle(img, (int(pred_sub_box[0]), int(pred_sub_box[1])),
        #                         (int(pred_sub_box[2]), int(pred_sub_box[3])), (0, 0, 255), 2)  # Blue: Predicted Subject
        #
        #     # 显示或保存图像
        #     plt.imshow(img)
        #     plt.title("Green: GT Object, Yellow: GT Subject, Red: Pred Object, Blue: Pred Subject")
        #     plt.savefig(f"visualization_{t['id']}.png")  # 保存图片
        #     plt.show()
        #
        #     break  # 只显示或保存一个样本
        # 扩展预测和 Ground Truth 数据
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    metric_logger.synchronize_between_processes()

    # 检查是否有 'id'
    if 'id' in gts[0]:
        img_ids = [img_gts['id'] for img_gts in gts]
        _, indices = np.unique(img_ids, return_index=True)
        preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
        gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    else:
        print("Warning: 'id' not found in gts. Skipping ID-based filtering.")

    if dataset_file in ['hico', 'hico1']:
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    if 'hico' in dataset_file and 'u' in dataset_file:
        evaluator = HICOUAEvaluator(preds, gts, data_loader.dataset.seen_triplets,
                                    data_loader.dataset.unseen_triplets, data_loader.dataset.correct_mat, args=args)
    if dataset_file in ['vcoco']:
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)
    if dataset_file in ['vcoco1']:
        evaluator = VCOCOPse(preds, gts, data_loader.dataset.correct_mat, args=args)
    if dataset_file in ['hvco'] and args.eval:
        print("Using HVCOEvaluator")
        evaluator = HVCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)
    if dataset_file == 'rice_cooker':
        print("Using RiceCookerEvaluator")
        evaluator = RiceCookerEvaluator(preds, gts, correct_mat=None)
    elif dataset_file in ['hvco']:
        print("Using HICOEvaluator (hvco fallback)")
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)

    stats = evaluator.evaluate()
    print(f"Validation mAP: {stats.get('mAP', 0.0):.4f}")
    print(f"Args received in evaluate_hoi: {args}")

    return stats

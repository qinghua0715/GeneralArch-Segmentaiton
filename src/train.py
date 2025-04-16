from typing import Dict, Union, List
from pathlib import Path
import torch
import csv
import pandas as pd
import imageio.v2 as imageio
from tqdm import tqdm
from utils.metric import compute_metrics
from utils.utils1 import AverageMeter
from collections import OrderedDict


def train(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    losses = AverageMeter()
    model.train()
    for image, label in tqdm(train_loader, total=len(train_loader), desc="Training"):
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), image.shape[0])
    return {'loss': losses.avg}


def val(model, val_loader, out_channels, criterion, device):
    """验证一个 epoch"""
    losses = AverageMeter()
    ious, dices = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        for image, label in tqdm(val_loader, total=len(val_loader), desc="Validating"):
            image, label = image.to(device), label.to(device)
            output = model(image)
            losses.update(criterion(output, label).item(), image.shape[0])
            metrics = compute_metrics(output, label, out_channels)
            ious.update(metrics['iou'], image.shape[0])
            dices.update(metrics['dice'], image.shape[0])
    return {'loss': losses.avg, 'iou': ious.avg, 'dice': dices.avg}


def test(model, args, test_loader, out_channels, device):
    """测试模型，batch_size=1"""
    # 初始化 AverageMeter
    if out_channels == 1:
        meters = {'iou': AverageMeter(), 'dice': AverageMeter(), 'recall': AverageMeter(),
                  'acc': AverageMeter(), 'rvd': AverageMeter()}
    else:
        num_foreground = out_channels - 1
        meters = {f'{key}_class{cls}': AverageMeter() for cls in range(1, num_foreground + 1)
                  for key in ['iou', 'dice', 'recall', 'acc', 'rvd']}

    save_dir = Path(args.save_path) / args.data / args.id / f'test_result-{args.time}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV 列
    if out_channels == 1:
        columns = ['num', 'image_name', 'iou', 'dice', 'recall', 'ACC', 'RVD']
    else:
        columns = ['num', 'image_name']
        for cls in range(1, out_channels):
            columns.extend([f'iou_class{cls}', f'dice_class{cls}', f'recall_class{cls}',
                            f'ACC_class{cls}', f'RVD_class{cls}'])

    df = pd.DataFrame(columns=columns)
    df.to_csv(save_dir.parent / f'test_acc-{args.time}.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

    model.eval()
    with torch.no_grad():
        for i, (name, image, label) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            name = name[0]  # batch_size=1
            image, label = image.to(device), label.to(device)
            output = model(image)
            metrics = compute_metrics(output, label, out_channels)

            # 确保数据类型
            i = int(i)  # 确保 num 为整数
            name = str(name)  # 确保 image_name 为字符串

            # 保存指标
            if out_channels == 1:
                # 单目标
                data = [i, name,
                        float(metrics['metrics_list'][0]['iou']),
                        float(metrics['metrics_list'][0]['dice']),
                        float(metrics['metrics_list'][0]['recall']),
                        float(metrics['metrics_list'][0]['acc']),
                        float(metrics['metrics_list'][0]['rvd'])]
                pd.DataFrame([data], columns=columns).to_csv(
                    save_dir.parent / f'test_acc-{args.time}.csv', mode='a', header=False, index=False,
                    quoting=csv.QUOTE_NONE, escapechar='\\'
                )
                # 更新 AverageMeter
                for key in ['iou', 'dice', 'recall', 'acc', 'rvd']:
                    meters[key].update(metrics['metrics_list'][0][key], 1)
            else:
                # 多目标
                data = [i, name]
                for cls_metrics in metrics['metrics_list'][0]:
                    data.extend([float(cls_metrics['iou']), float(cls_metrics['dice']),
                                 float(cls_metrics['recall']), float(cls_metrics['acc']),
                                 float(cls_metrics['rvd'])])
                pd.DataFrame([data], columns=columns).to_csv(
                    save_dir.parent / f'test_acc-{args.time}.csv', mode='a', header=False, index=False,
                    quoting=csv.QUOTE_NONE, escapechar='\\'
                )
                # 更新 AverageMeter
                for cls_idx, cls_metrics in enumerate(metrics['metrics_list'][0], 1):
                    for key in ['iou', 'dice', 'recall', 'acc', 'rvd']:
                        meters[f'{key}_class{cls_idx}'].update(cls_metrics[key], 1)

            # 保存预测结果，单目标分割保存成黑白图像，多目标分类保存结果肉眼全黑是由于其值为0，1，2，3都太小，需要单独进行可视化，比如将0定义成黑色，1定义成红色等
            preds = (output.softmax(1).argmax(1) if out_channels > 1 else (output.sigmoid() > 0.5).float())
            preds = preds.squeeze().cpu().numpy()
            imageio.imwrite(
                save_dir / f"{name}.png",
                (preds * 255.0).astype('uint8') if out_channels == 1 else preds.astype('uint8')
            )
    # 返回平均指标
    if out_channels == 1:
        avg_metrics = {key: meter.avg for key, meter in meters.items()}
    else:
        # 计算前景类均值
        avg_metrics = {'iou': 0.0, 'dice': 0.0, 'recall': 0.0, 'acc': 0.0, 'rvd': 0.0}
        num_foreground = out_channels - 1
        for cls in range(1, out_channels):
            for key in ['iou', 'dice', 'recall', 'acc', 'rvd']:
                avg_metrics[key] += meters[f'{key}_class{cls}'].avg / num_foreground
                avg_metrics[f'{key}_class{cls}'] = meters[f'{key}_class{cls}'].avg  # 保留逐类均值
    return avg_metrics

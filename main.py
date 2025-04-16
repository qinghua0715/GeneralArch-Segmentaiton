import datetime
import torch
import torch.nn as nn
import os
import json
import pandas as pd
import shutil
from pathlib import Path
from utils.GenericDataset import GenericDataset
from src.models import model_registry
from src.train import train, val, test
from utils.utils1 import load_config, parse_args, setup_logging
from utils.transform import dataset_transform
from utils.loss import BceDiceLoss, CrossEntropyDiceLoss


def load_split(data_dir):
    """从 split.json 中加载训练、验证和测试路径列表"""
    split_file = os.path.join(data_dir, 'split.json')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"未找到数据集划分文件: {split_file}. 请先运行预处理脚本生成 split.json")
    with open(split_file, 'r') as f:
        split_dict = json.load(f)
    return (split_dict['train_images'], split_dict['train_labels'],
            split_dict['val_images'], split_dict['val_labels'],
            split_dict['test_images'], split_dict['test_labels'])


def save_gold_standard(data_dir, dataset_name, test_image_paths, test_label_paths, gold_standard_dir):
    """保存测试集的金标准图像和标签"""
    output_dir = Path(gold_standard_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / 'image'
    label_dir = output_dir / 'label'
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    # 检查现有文件是否与测试集路径完全匹配
    expected_image_names = {os.path.basename(p) for p in test_image_paths}
    expected_label_names = {os.path.basename(p) for p in test_label_paths}
    current_image_names = set(os.listdir(image_dir)) if image_dir.exists() else set()
    current_label_names = set(os.listdir(label_dir)) if label_dir.exists() else set()

    # 如果文件名集合不完全一致，则重新保存
    if expected_image_names != current_image_names or expected_label_names != current_label_names:
        print(f"金标准文件名不匹配，正在重新保存至 {output_dir}")
        # 清空现有目录（防止残留无关文件）
        for fname in current_image_names:
            (image_dir / fname).unlink(missing_ok=True)
        for fname in current_label_names:
            (label_dir / fname).unlink(missing_ok=True)

        # 复制文件
        for img_path, lbl_path in zip(test_image_paths, test_label_paths):
            try:
                img_name = os.path.basename(img_path)
                lbl_name = os.path.basename(lbl_path)
                shutil.copy(img_path, image_dir / img_name)
                shutil.copy(lbl_path, label_dir / lbl_name)
            except Exception as e:
                print(f"保存金标准失败 {img_path}: {e}")
        print(f"金标准保存完成：{len(test_image_paths)} 张图像和标签")
    else:
        print(f"金标准已存在且匹配，跳过保存")


def main(args):
    # 加载数据集配置
    config = load_config("config/datasets.yaml")
    dataset_config = config['datasets'][args.data]

    # 设置路径参数（仅用于保存结果）
    args.gold_standard_dir = config['paths']['gold_standard']
    args.save_path = config['paths']['save_model']

    # 加载预生成的训练、验证和测试路径列表
    data_dir = dataset_config['path']
    train_image_paths, train_label_paths, val_image_paths, val_label_paths, test_image_paths, test_label_paths = load_split(
        data_dir)

    # 保存金标准
    save_gold_standard(data_dir, args.data, test_image_paths, test_label_paths, args.gold_standard_dir)

    # 设置类别数
    if args.num_classes > 1:
        out_channels = args.num_classes + 1
        print("当前为多目标分割任务")
    else:
        out_channels = args.num_classes
        print("当前为单目标分割任务")

    # 创建数据集
    train_dataset = GenericDataset(
        image_paths=train_image_paths,
        label_paths=train_label_paths,
        train_type='train',
        image_size=(224, 224),
        transform=lambda x, t, s: dataset_transform(
            x, t, s, out_channels=out_channels
        )
    )
    val_dataset = GenericDataset(
        image_paths=val_image_paths,
        label_paths=val_label_paths,
        train_type='val',
        image_size=(224, 224),
        transform=lambda x, t, s: dataset_transform(
            x, t, s, out_channels=out_channels
        )
    )
    test_dataset = GenericDataset(
        image_paths=test_image_paths,
        label_paths=test_label_paths,
        train_type='test',
        image_size=(224, 224),
        transform=lambda x, t, s: dataset_transform(
            x, t, s, out_channels=out_channels
        )
    )

    # 创建数据加载器
    num_workers = min([os.cpu_count() - 2, args.batch_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    # 初始化设备和模型
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"使用 {num_gpus} 个GPU(s)")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("使用 CPU")

    model = model_registry.get(args.id, in_channels=3, out_channels=out_channels)
    if num_gpus > 1:
        model = nn.DataParallel(model)  # 包装为 DataParallel
    model = model.to(device)

    # 初始化日志和记录文件
    setup_logging(args)
    # 打印模型信息
    print(f"Network: {args.id}")
    if num_gpus > 1:
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}")
        print(f"Network Architecture of Model {args.id}:")
        print(model.module)
    else:
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Network Architecture of Model {args.id}:")
        print(model)

    # 初始化优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-6, verbose=True
    )
    if out_channels == 1:
        criterion = BceDiceLoss()
    else:
        weights = torch.tensor([0.1] + [1.0] * (out_channels - 1)).to(device)    # weights=[0.1, 1.0, 1.0,..]：背景权重低。
        criterion = CrossEntropyDiceLoss(wc=0.5, wd=1.0, class_weights=weights)    # wc=0.5, wd=1.0：Dice 更重要;
    criterion = criterion.to(device)  # 确保损失函数在 GPU

    # 初始化记录文件
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice'])
    df.to_csv(Path(args.save_path) / args.data / args.id / f'train_val_acc-{args.time}.csv', index=False)

    # 断点继续训练
    checkpoint_path = Path(args.save_path) / args.data / args.id / "checkpoint.pth"
    start_epoch = 0
    best_iou = 0
    trigger = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        if num_gpus > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint['best_iou']
        trigger = checkpoint['trigger']
        print(f"Resumed from checkpoint at epoch {start_epoch}, best IoU: {best_iou:.4f}")
    else:
        print("Starting training from scratch")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}/{args.epochs - 1}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        train_metric = train(model, train_loader, optimizer, criterion, device)
        val_metric = val(model, val_loader, out_channels, criterion, device)
        lr_scheduler.step(val_metric['iou'])

        print(f"Train loss: {train_metric['loss']:.4f}, Val loss: {val_metric['loss']:.4f}, "
              f"Val IoU: {val_metric['iou']:.4f}, Val Dice: {val_metric['dice']:.4f}")

        # 记录指标
        data = [epoch, train_metric['loss'], val_metric['loss'], val_metric['iou'], val_metric['dice']]
        pd.DataFrame([data]).to_csv(
            Path(args.save_path) / args.data / args.id / f'train_val_acc-{args.time}.csv',
            mode='a', header=False, index=False
        )

        # 保存最佳模型
        if val_metric['iou'] > best_iou:
            best_iou = val_metric['iou']
            trigger = 0
            model_path = Path(args.save_path) / args.data / args.id
            model_path.mkdir(parents=True, exist_ok=True)
            # 保存原始模型权重
            state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(state_dict, model_path / "best_model.pth")
            print(f"-------> Saved best model with IoU: {best_iou:.4f}")
            save_checkpoint = True
        else:
            trigger += 1

        # 定期保存检查点（每 args.checkpoint_freq epoch）
        if args.checkpoint_freq > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint = True

        if save_checkpoint:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_iou': best_iou,
                'trigger': trigger
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}")

        if trigger >= args.early_stop:
            print("Early stopping triggered!")
            break

    # 测试
    # 恢复原始模型（单 GPU 推理）
    if num_gpus > 1:
        model = model.module
    model = model.to(device)
    # 加载最佳模型
    state_dict = torch.load(Path(args.save_path) / args.data / args.id / "best_model.pth")
    model.load_state_dict(state_dict)
    test_metric = test(model, args, test_loader, out_channels, device)
    print(f"Test IoU: {test_metric['iou']:.4f}, Dice: {test_metric['dice']:.4f}, "
          f"Recall: {test_metric['recall']:.4f}, ACC: {test_metric['acc']:.4f}, RVD: {test_metric['rvd']:.4f}")


if __name__ == "__main__":
    args = parse_args("config/experiment.yaml")
    args.time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main(args)

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """适用于二分类的 Dice 损失"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # [batch_size, 1, H, W]
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class MultiClassDiceLoss(nn.Module):
    """适用于多分类的 Dice 损失"""
    def __init__(self, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: 模型的输出，形状为 (N, C, H, W)，未经过Softmax
        targets: 真实标签，形状为 (N, H, W)，值为类别索引
        """
        inputs = F.softmax(inputs, dim=1)  # [N, C, H, W]
        N, C, H, W = inputs.size()
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()  # [N, C, H, W]

        inputs_flat = inputs.view(N, C, -1)
        targets_flat = targets_one_hot.view(N, C, -1)

        dice_loss = 0
        for c in range(1, C):  # 跳过背景
            intersection = (inputs_flat[:, c] * targets_flat[:, c]).sum(-1)
            union = inputs_flat[:, c].sum(-1) + targets_flat[:, c].sum(-1)
            dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice_score
        dice_loss = dice_loss.mean() / (C - 1)
        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, label):
        if pred.shape[1] != 1 or label.shape[1] != 1:
            raise ValueError(f"二分类预测和标签应为单通道，得到 pred: {pred.shape}, label: {label.shape}")
        bceloss = self.bce(pred, label)
        diceloss = self.dice(pred, label)
        return self.wb * bceloss + self.wd * diceloss


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, wc=1, wd=1, class_weights=None):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = MultiClassDiceLoss()
        self.wc = wc
        self.wd = wd

    def forward(self, inputs, targets):
        if inputs.shape[1] <= 2 or targets.dim() != 3:
            raise ValueError(f"多分类预测应为多通道，标签为 [B, H, W]，得到 inputs: {inputs.shape}, targets: {targets.shape}")
        if targets.max() >= inputs.shape[1] or targets.min() < 0:
            raise ValueError(f"标签值超出范围: min={targets.min()}, max={targets.max()}, 应为 [0, {inputs.shape[1] - 1}]")

        celoss = self.ce_loss(inputs, targets)
        diceloss = self.dice_loss(inputs, targets)
        return self.wc * celoss + self.wd * diceloss


# 示例用法
if __name__ == "__main__":
    # 假设有一个模型输出和对应的真实标签
    N, C, H, W = 2, 3, 256, 256  # 批量大小、类别数、高度、宽度
    inputs = torch.randn(N, C, H, W)  # 模型的原始输出，未经过Softmax
    targets = torch.randint(0, C, (N, H, W))  # 真实标签，每个像素的类别索引

    # 创建组合损失函数的实例
    criterion = CrossEntropyDiceLoss(1, 1)
    # 计算损失
    loss = criterion(inputs, targets)
    print(f"Total Loss: {loss.item():.4f}")

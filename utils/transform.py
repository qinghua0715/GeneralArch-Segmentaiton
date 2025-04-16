import random
import numpy as np
import torch
import torchvision.transforms as ts
import torchvision.transforms.functional as TF


class RandomCrop(object):
    """随机裁剪图像和标签，使用 torchvision 的 RandomCrop 实现"""

    def __init__(self, size, padding=0, pad_if_needed=False):
        """
        初始化随机裁剪变换

        参数:
            size (序列或整数): 裁剪的目标尺寸。如果是整数，则裁剪为正方形。
            padding (整数或序列): 每边填充的大小，默认 0。
            pad_if_needed (布尔值): 如果图像小于目标尺寸，是否自动填充。
        """
        self.transform = ts.RandomCrop(size, padding=padding, pad_if_needed=pad_if_needed)

    def __call__(self, img, lab):
        """
        执行裁剪操作

        参数:
            img (Tensor): 要裁剪的图像张量
            lab (Tensor): 要裁剪的标签张量

        返回:
            Tensor: 裁剪后的图像和标签张量
        """
        seed = torch.randint(0, 2 ** 32, (1,)).item()
        torch.manual_seed(seed)
        img = self.transform(img)
        torch.manual_seed(seed)
        lab = self.transform(lab)
        return img, lab


def randomflip_rotate(img, lab, p=0.5, degrees=30, multiclass=False):
    """
    对图像和标签应用随机水平/垂直翻转和旋转

    参数:
        img (Tensor): 图像张量，形状 [C, H, W]
        lab (Tensor): 标签张量，二分类为 [1, H, W]，多分类为 [H, W]
        p (float): 应用翻转的概率
        degrees (float 或 tuple): 旋转角度范围
        multiclass (bool): 是否为多分类任务

    返回:
        Tensor: 变换后的图像和标签张量
    """
    # 随机水平翻转
    if random.random() < p:
        img = torch.flip(img, dims=[-1])
        lab = torch.flip(lab, dims=[-1])

    # 随机垂直翻转
    if random.random() < p:
        img = torch.flip(img, dims=[-2])
        lab = torch.flip(lab, dims=[-2])

    # 随机旋转
    if degrees:
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        angle = random.uniform(degrees[0], degrees[1])

        # 为多分类标签添加临时通道维度
        lab_orig_shape = lab.shape
        if multiclass and len(lab_orig_shape) == 2:
            lab = lab.unsqueeze(0)  # [H, W] -> [1, H, W]

        # 应用旋转
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        lab = TF.rotate(lab, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 恢复多分类标签形状
        if multiclass and len(lab_orig_shape) == 2:
            lab = lab.squeeze(0)  # [1, H, W] -> [H, W]

    return img, lab


def dataset_transform(sample, train_type, image_size=(224, 224), out_channels=1, flip_prob=0.5, rotate_degrees=30):
    """
    对图像和标签进行数据增强和预处理，适用于分割任务

    参数:
        sample (dict): 包含 'image' (numpy 数组) 和 'label' (numpy 数组) 的字典
        train_type (str): 数据类型 ('train', 'val', 'test')
        image_size (tuple): 目标尺寸，默认为 (224, 224)
        multiclass (bool): 如果为 True，标签处理为多分类（long 张量）；否则为二分类（灰度图）
        flip_prob (float): 翻转概率，默认为 0.5
        rotate_degrees (float): 最大旋转角度，默认为 30

    返回:
        dict: 包含变换后的图像和标签张量的字典
    """
    image, label = sample['image'], sample['label']

    # 检查输入通道数
    in_channels = image.shape[-1]
    if in_channels != 3:
        raise ValueError(f"图像通道数 {in_channels} 不支持，仅支持3通道")

    # 将 numpy 数组转为张量
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC 转为 CHW，并归一化到 [0,1]

    # 处理标签
    multiclass = out_channels > 2
    if multiclass:
        label = torch.from_numpy(label).long()  # 多分类：转为 long 张量，形状 [H, W]
        # 检查标签值范围
        if label.max() >= out_channels or label.min() < 0:
            raise ValueError(f"多分类标签值超出范围: min={label.min()}, max={label.max()}, 应为 [0, {out_channels - 1}]")
    else:
        label = torch.from_numpy(label).float().unsqueeze(0)  # 二分类：转为 float 张量，形状 [1, H, W]
        # 可选：检查二分类标签值（通常为 0 或 1）
        if label.max() > 1 or label.min() < 0:
            label = (label > 0).float()  # 强制转为 0 或 1

    # 训练阶段应用数据增强
    if train_type == 'train':
        # 随机裁剪
        crop = RandomCrop(size=image_size, pad_if_needed=True)
        image, label = crop(image, label)

        # 随机翻转和旋转
        image, label = randomflip_rotate(image, label, p=flip_prob, degrees=rotate_degrees, multiclass=multiclass)

    # 如果尺寸不匹配，调整大小（验证/测试阶段）
    if image.shape[-2:] != image_size:
        image = TF.resize(image, image_size, interpolation=TF.InterpolationMode.BILINEAR)  # 图像双线性插值

        lab_orig_shape = label.shape
        if multiclass and len(lab_orig_shape) == 2:
            label = label.unsqueeze(0)
        label = TF.resize(label, image_size, interpolation=TF.InterpolationMode.NEAREST)  # 标签最近邻插值
        if multiclass and len(lab_orig_shape) == 2:
            label = label.squeeze(0)


    # 图像归一化
    normalize = ts.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = normalize(image)

    return {'image': image, 'label': label}

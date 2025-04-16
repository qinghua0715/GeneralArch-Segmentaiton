import imageio.v2 as imageio
import numpy as np
import os
from torch.utils.data.dataset import Dataset


class GenericDataset(Dataset):
    def __init__(self, image_paths, label_paths, train_type='train', image_size=(224, 224), transform=None):
        """
        通用的数据集类，接受图像和标签的完整路径列表。

        Args:
            image_paths (list): 图像文件的完整路径列表
            label_paths (list): 标签文件的完整路径列表
            train_type (str): 数据类型 ('train', 'val', 'test')
            image_size (tuple): 图像目标大小，默认为 (224, 224)
            transform (callable, optional): 数据增强或预处理变换
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.train_type = train_type
        self.image_size = image_size
        self.transform = transform

        # 验证输入
        if self.train_type not in ['train', 'val', 'test']:
            raise ValueError("train_type 必须是 'train', 'val' 或 'test'")
        if len(self.image_paths) != len(self.label_paths):
            raise ValueError(f"图像路径数量 ({len(self.image_paths)}) 和标签路径数量 ({len(self.label_paths)}) 不匹配")

    def __getitem__(self, item: int):
        """获取单个样本"""
        image_path = self.image_paths[item]
        label_path = self.label_paths[item]

        # 根据文件扩展名加载图像和标签
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        elif image_path.endswith('.png'):
            image = imageio.imread(image_path)
        else:
            raise ValueError(f"不支持的文件格式: {image_path}")

        if label_path.endswith('.npy'):
            label = np.load(label_path)
        elif label_path.endswith('.png'):
            label = imageio.imread(label_path)
        else:
            raise ValueError(f"不支持的文件格式: {label_path}")

        # 提取文件名（用于测试模式返回）
        name = os.path.basename(image_path)

        # 创建样本字典
        sample = {'image': image, 'label': label}

        # 应用变换（如果提供）
        if self.transform is not None:
            sample = self.transform(sample, self.train_type, self.image_size)

        # 根据 train_type 返回数据
        if self.train_type == 'test':
            return name, sample['image'], sample['label']
        return sample['image'], sample['label']

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

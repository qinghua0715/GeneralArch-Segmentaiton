import os
import json
from sklearn.model_selection import train_test_split
from pathlib import Path


def generate_isic2018_lists(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    image_dir = os.path.join(data_dir, 'image')
    label_dir = os.path.join(data_dir, 'label')

    # 获取完整路径
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    label_paths = sorted([os.path.join(label_dir, fname) for fname in os.listdir(label_dir)])

    if len(image_paths) != len(label_paths):
        raise ValueError("图像和标签数量不匹配")

    # 划分数据集索引
    indices = list(range(len(image_paths)))
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

    # 生成完整路径列表
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [label_paths[i] for i in val_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_labels = [label_paths[i] for i in test_idx]

    # 保存为 JSON 文件
    split_dict = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels
    }
    output_file = os.path.join('./ISIC2018_png_224', 'split.json')
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

    print(f"ISIC2018 数据集划分完成，保存至 {output_file}")


def generate_busi_lists(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # 定义图像和标签目录路径
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'masks')

    # 检查输入目录是否存在
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录 {image_dir} 不存在。")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"标签目录 {label_dir} 不存在。")

    # 获取图像和标签文件路径，仅包括指定扩展名的文件
    valid_extensions = ('.png', '.jpg')  # 有效图像扩展名
    image_paths = sorted([
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(image_dir, fname))
    ])
    label_paths = sorted([
        os.path.join(label_dir, fname) for fname in os.listdir(label_dir)
        if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(label_dir, fname))
    ])

    # 验证图像和标签数量是否匹配
    if len(image_paths) != len(label_paths):
        raise ValueError(f"图像数量 ({len(image_paths)}) 和标签数量 ({len(label_paths)}) 不匹配。")

    # 划分数据集索引
    indices = list(range(len(image_paths)))
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

    # 生成训练、验证和测试集的路径列表
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [label_paths[i] for i in val_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_labels = [label_paths[i] for i in test_idx]

    # 创建用于JSON输出的字典
    split_dict = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels
    }

    # 定义并创建输出目录
    save_dir = f'./BUSI_png_224'
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'split.json'

    # 确保输出路径不是目录
    if output_file.is_dir():
        raise IsADirectoryError(f"输出路径 {output_file} 是一个目录，而非文件。")

    # 将划分结果保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

    # 打印完成信息
    print(f"BUSI数据集划分完成，保存至 {output_file}")


def generate_kvasir_lists(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    # 定义图像和标签目录路径
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'masks')

    # 检查输入目录是否存在
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录 {image_dir} 不存在。")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"标签目录 {label_dir} 不存在。")

    # 获取图像和标签文件路径，仅包括指定扩展名的文件
    valid_extensions = ('.png', '.jpg')  # 有效图像扩展名
    image_paths = sorted([
        os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
        if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(image_dir, fname))
    ])
    label_paths = sorted([
        os.path.join(label_dir, fname) for fname in os.listdir(label_dir)
        if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(label_dir, fname))
    ])

    # 验证图像和标签数量是否匹配
    if len(image_paths) != len(label_paths):
        raise ValueError(f"图像数量 ({len(image_paths)}) 和标签数量 ({len(label_paths)}) 不匹配。")

    # 划分数据集索引
    indices = list(range(len(image_paths)))
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

    # 生成训练、验证和测试集的路径列表
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [label_paths[i] for i in val_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_labels = [label_paths[i] for i in test_idx]

    # 创建用于JSON输出的字典
    split_dict = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels
    }

    # 定义并创建输出目录
    save_dir = f'./Kvasir_png_224'
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'split.json'

    # 确保输出路径不是目录
    if output_file.is_dir():
        raise IsADirectoryError(f"输出路径 {output_file} 是一个目录，而非文件。")

    # 将划分结果保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

    # 打印完成信息
    print(f"BUSI数据集划分完成，保存至 {output_file}")


def generate_glas_lists(data_dir):
    """
    为Glas_png_224数据集生成训练、验证和测试集的路径列表，并保存为JSON文件。
    测试集使用test文件夹，训练和验证集从train文件夹按8:2划分。
    通过文件名核心部分（去除_anno后缀）配对图像和标签，确保一一对应。

    参数:
        data_dir (str): 数据集根目录路径
    """
    # 定义train和test的images和masks目录
    train_image_dir = os.path.join(data_dir, 'train', 'images')
    train_mask_dir = os.path.join(data_dir, 'train', 'masks')
    test_image_dir = os.path.join(data_dir, 'test', 'images')
    test_mask_dir = os.path.join(data_dir, 'test', 'masks')

    # 检查所有目录是否存在
    for directory in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录 {directory} 不存在。")

    # 定义有效图像扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # 获取train文件夹中的图像和标签文件名
    train_image_files = [fname for fname in os.listdir(train_image_dir)
                         if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(train_image_dir, fname))]
    train_mask_files = [fname for fname in os.listdir(train_mask_dir)
                        if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(train_mask_dir, fname))]

    # 获取test文件夹中的图像和标签文件名
    test_image_files = [fname for fname in os.listdir(test_image_dir)
                        if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(test_image_dir, fname))]
    test_mask_files = [fname for fname in os.listdir(test_mask_dir)
                       if fname.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(test_mask_dir, fname))]

    # 提取文件名核心部分（去除扩展名和_anno后缀）
    def get_core_name(filename):
        """提取文件名核心部分，去除扩展名和_anno后缀"""
        base = os.path.splitext(filename)[0]
        return base.replace('_anno', '')

    # 配对train文件夹中的图像和标签
    train_image_paths = []
    train_mask_paths = []
    train_image_core_names = {get_core_name(fname): fname for fname in train_image_files}
    for mask_file in train_mask_files:
        core_name = get_core_name(mask_file)
        if core_name in train_image_core_names:
            train_image_paths.append(os.path.join(train_image_dir, train_image_core_names[core_name]))
            train_mask_paths.append(os.path.join(train_mask_dir, mask_file))
        else:
            print(f"警告: 标签文件 {mask_file} 没有对应的图像文件，跳过。")

    # 配对test文件夹中的图像和标签
    test_image_paths = []
    test_mask_paths = []
    test_image_core_names = {get_core_name(fname): fname for fname in test_image_files}
    for mask_file in test_mask_files:
        core_name = get_core_name(mask_file)
        if core_name in test_image_core_names:
            test_image_paths.append(os.path.join(test_image_dir, test_image_core_names[core_name]))
            test_mask_paths.append(os.path.join(test_mask_dir, mask_file))
        else:
            print(f"警告: 标签文件 {mask_file} 没有对应的图像文件，跳过。")

    # 验证train和test中的图像和标签数量是否匹配
    if len(train_image_paths) != len(train_mask_paths):
        raise ValueError(f"训练集图像数量 ({len(train_image_paths)}) 和标签数量 ({len(train_mask_paths)}) 不匹配。")
    if len(test_image_paths) != len(test_mask_paths):
        raise ValueError(f"测试集图像数量 ({len(test_image_paths)}) 和标签数量 ({len(test_mask_paths)}) 不匹配。")

    # 对train数据按8:2划分训练和验证集
    train_indices = list(range(len(train_image_paths)))
    train_idx, val_idx = train_test_split(train_indices, train_size=0.8, random_state=42)

    # 生成训练和验证集路径
    train_images = [train_image_paths[i] for i in train_idx]
    train_masks = [train_mask_paths[i] for i in train_idx]
    val_images = [train_image_paths[i] for i in val_idx]
    val_masks = [train_mask_paths[i] for i in val_idx]

    # 测试集直接使用test文件夹数据
    test_images = test_image_paths
    test_masks = test_mask_paths

    # 创建JSON字典
    split_dict = {
        "train_images": train_images,
        "train_masks": train_masks,
        "val_images": val_images,
        "val_masks": val_masks,
        "test_images": test_images,
        "test_masks": test_masks
    }

    # 定义输出路径并创建目录
    save_dir = f'./Glas_png_224'
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / 'split.json'

    # 确保输出路径不是目录
    if output_file.is_dir():
        raise IsADirectoryError(f"输出路径 {output_file} 是一个目录，而非文件。")

    # 保存JSON文件
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

    print(f"Glas数据集划分完成，保存至 {output_file}")


def generate_ACDC_lists(data_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    image_dir = os.path.join(data_dir, 'image')
    label_dir = os.path.join(data_dir, 'label')

    # 获取完整路径
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    label_paths = sorted([os.path.join(label_dir, fname) for fname in os.listdir(label_dir)])

    if len(image_paths) != len(label_paths):
        raise ValueError("图像和标签数量不匹配")

    # 划分数据集索引
    indices = list(range(len(image_paths)))
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

    # 生成完整路径列表
    train_images = [image_paths[i] for i in train_idx]
    train_labels = [label_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_labels = [label_paths[i] for i in val_idx]
    test_images = [image_paths[i] for i in test_idx]
    test_labels = [label_paths[i] for i in test_idx]

    # 保存为 JSON 文件
    split_dict = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels
    }
    output_file = os.path.join('./ACDC_png_224', 'split.json')
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

    print(f"ACDC 数据集划分完成，保存至 {output_file}")


if __name__ == "__main__":
    # generate_isic2018_lists('/home/dell/Code/data/ISIC2018_png_224')
    # generate_busi_lists('/home/dell/Code/data/BUSI_png_224')
    # generate_ACDC_lists('/home/dell/Code/data/ACDC_png_224')
    # generate_kvasir_lists('/home/dell/Code/data/Kvasir_png_224')
    generate_glas_lists('/home/dell/Code/data/Glas_png_224')
    # generate_glas_lists('./data/Glas_png_224')
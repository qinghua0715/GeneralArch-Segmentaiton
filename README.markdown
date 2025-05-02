# GeneralArch-Segmentation

GeneralArch-Segmentation 是一个用于医学图像分割的通用训练框架，旨在方便切换模型、添加新数据集，并支持单目标和多目标分割任务。它能够一键完成模型的训练、验证和测试，并保存相关结果，适用于医学图像分割的研究和开发。

---

## 项目概述

- **目标**: 提供一个灵活、可扩展的医学图像分割训练框架。
- **功能**:
  - 支持动态切换模型。
  - 支持添加新数据集。
  - 支持单目标和多目标分割。
  - 一键完成训练、验证和测试，并保存结果。

---

## 目录结构

```
GeneralArch-Segmentation1
├── config
│   ├── datasets.yaml       # 配置数据集路径和相关信息
│   └── experiment.yaml     # 配置实验参数（如学习率、模型ID等）
├── data                    # 存放数据集或数据列表的json文件
├── Models                  # 存放不同模型的定义
├── save_model              # 保存模型训练和测试结果
│   ├── test_result         # 测试结果图像
│   ├── best_model.pth      # 最优模型权重
│   ├── checkpoint.pth      # 检查点文件
│   ├── Log                 # 日志文件
│   ├── test_acc            # 测试指标
│   └── train_val_acc       # 训练和验证过程中的loss变化
├── segment_result          # 保存测试集的金标准数据
├── scr
│   ├── moedels.py          # 动态加载模型（可能是models.py的拼写错误）
│   ├── train.py            # 包含训练、验证和测试函数
│   └── utils
│       ├── GenericDataset.py  # 数据集加载器
│       ├── loss.py         # 损失函数定义
│       ├── metric.py       # 指标计算
│       ├── transformer.py  # 数据预处理
│       └── utils1.py       # 日志、工具函数等
├── main.py                 # 主函数，程序入口
```

---

## 安装依赖

确保您的环境已安装以下依赖：

- Python 3.10+
- PyTorch
- pandas
- shutil
- pathlib

安装依赖示例：

```bash
pip install torch pandas shutil pathlib
```

---

## 配置

### 1. 数据集配置

- 文件: `config/datasets.yaml`
- 作用: 定义数据集路径和其他相关信息。
- 示例:

  ```yaml
  datasets:
    ISIC2018_png_224:
      path: "./data/ISIC2018_png_224"

    ACDC_png_224:
      path: "./data/ACDC_png_224"

  paths:
    gold_standard: "./segment_result/gold_standard"
    save_model: "./save_model"
  ```

### 2. 实验参数配置

- 文件: `config/experiment.yaml`
- 作用: 定义训练参数，如模型ID、学习率等。
- 示例:

  ```yaml
  experiment:
    id: "S_Net"
    data: "ISIC2018_png_224"
    num_classes: 1
    n_splits: 5
    save_path: "./save_model"
    epochs: 120
    resume: False
    checkpoint_freq: 10
    early_stop: 10
    batch_size: 52
    lr: 0.001
    weight_decay: 0.0001
  ```

---

## 数据集准备

1. **数据集放置**:

   - 将数据集放入 `data` 目录，例如 `data/my_dataset`。

2. **生成 split.json**:

   - 在数据集目录下创建 `split.json`，包含训练、验证和测试的图像及标签的完成路径列表。这一步可以用AI帮你完成，告诉AI你的数据集目录结构，文件命名方式和数据集结构，让其按照指定比例划分数据集生成包含完整路径列表的json文件。
   - 示例格式:

     ```json
     {
         "train_images": ["data/my_dataset/images/train1.png", "..."],
         "train_labels": ["data/my_dataset/labels/train1.png", "..."],
         "val_images": ["data/my_dataset/images/val1.png", "..."],
         "val_labels": ["data/my_dataset/labels/val1.png", "..."],
         "test_images": ["data/my_dataset/images/test1.png", "..."],
         "test_labels": ["data/my_dataset/labels/test1.png", "..."]
     }
     ```

---

## 模型

- **存放位置**: `Models` 目录。
- **注册模型**: 在 `scr/models.py`中定义并注册模型，以便动态加载。
- 示例:

  ```python
  model_registry.register("Unet", "Models.Unet", "Unet")
  ```

---

## 使用方法

### 1. 配置参数

- 编辑 `config/datasets.yaml` 和 `config/experiment.yaml`，确保路径和参数正确。

### 2. 运行训练

- 使用以下命令启动训练、验证和测试：

  ```bash
  python main.py --data my_dataset --id my_model --num_classes 1 --batch_size 16 --lr 0.001 --epochs 50
  ```
- 或者在配置好参数后，在pycharm中直接运行main.py

### 3. 命令行参数

| 参数 | 描述 | 默认值    |
| --- | --- |--------|
| `--data` | 数据集名称 | 无      |
| `--id` | 模型ID | 无      |
| `--num_classes` | 类别数 | 1      |
| `--batch_size` | 批次大小 | 16     |
| `--lr` | 学习率 | 0.001  |
| `--weight_decay` | 权重衰减 | 0.0001 |
| `--epochs` | 训练轮数 | 50     |
| `--resume` | 是否从检查点恢复训练 | False  |
| `--checkpoint_freq` | 检查点保存频率 | 5      |
| `--early_stop` | 早停耐心 | 10     |

---

## 输出结果

- **最佳模型权重**: `save_model/my_dataset/my_model/best_model.pth`
- **检查点**: `save_model/my_dataset/my_model/checkpoint.pth`
- **训练验证指标**: `save_model/my_dataset/my_model/train_val_acc-{time}.csv`
- **测试结果图像**: `save_model/my_dataset/my_model/test_result`
- **测试指标**: `save_model/my_dataset/my_model/test_acc`
- **日志**: `save_model/my_dataset/my_model/Log`
- **金标准**: `segment_result/gold_standard/my_dataset/image` 和 `segment_result/gold_standard/my_dataset/label`

---

## 注意事项

- 确保 `split.json` 文件存在且路径正确，图像与标签配对。
- 根据任务类型设置 `--num_classes`（单目标为1，多目标大于1）。
- 多GPU训练会自动使用 `DataParallel`。
- 自定义预处理可在 `utils/transformer.py` 中实现。
- 新增模型需在 `Models` 目录定义并在 `scr/moedels.py` 中注册。
- 新增数据集需要在 `datasets.yaml`中增加。
- 新增损失函数或指标可在 `utils/loss.py` 和 `utils/metric.py` 中定义。

---

## 示例

训练一个单目标分割模型：

```bash
python main.py --data my_dataset --id my_model --num_classes 1 --batch_size 16 --lr 0.001 --epochs 50
```

---

## 贡献指南

- **报告问题**: 在 GitHub 上提交 issue。
- **其他**: 本代码为个人参考众多研究者的代码辅助AI进行编写，有关多卡并行的方式因设备限制并未完全测试，谨慎使用。另，如有错误或不足，欢迎指出，感谢！

---

## 许可证

- **使用范围**: 禁止商用，仅供个人或研究者互相学习使用，引用请注明来源。

---

## 联系方式

\[暂无\]

---

## 致谢

感谢所有为医学图像分割领域做出贡献的研究者和开发者。

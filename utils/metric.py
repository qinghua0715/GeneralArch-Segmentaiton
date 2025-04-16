import numpy as np
import torch
from typing import Dict, Union, List
import torch.nn.functional as F


# 用于计算混淆矩阵
def fast_hist(pred, label, n):
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    k = (label >= 0) & (pred < n)
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，斜对角线上的为分类正确的像素点
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return np.diag(hist) / (np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) + 1e-6)


def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


# 用混淆矩阵计算IOU
def compute_mIoU(pred, label, num_classes):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    # 计算混淆矩阵，输入要求展开成一维
    hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
    print(hist)
    #   计算所有验证集图片的逐类别mIoU值
    per_iou = per_class_iou(hist)
    mPA = per_class_PA(hist)
    # 在所有验证集图像上求所有类别平均的mIoU值，不包括背景部分(0)
    per_iou = per_iou[1:]
    mIoUs = np.nanmean(per_iou)
    mPA = np.nanmean(mPA)
    return mIoUs


# 一种计算iou和dice的方法
def iou_and_dice(output, target):
    smooth = 1e-6

    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()
        output = output.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = intersection / (union + smooth)
    # 根据dice和iou的关系计算dice
    dice = (2 * iou) / (iou + 1)
    return iou, dice


# 一种计算dice coef的方法，计算前先进行了一次sigmoid，使得计算出的dice更稳定，但与定义有差异
def dice_coef(output, target):
    smooth = 1e-6

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection) / \
        (output.sum() + target.sum() + smooth)


# 一种官方给出的计算dice的方法，适用于batch为1的时候用
def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = result > 0.5
    reference = reference > 0.5
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / (float(size_i1 + size_i2) + 1e-6)
    except ZeroDivisionError:
        dc = 0.0

    return dc


# 一种基于官方方法改进的计算dice的方法，可以计算batch≥1时的平均dice
def dc_mean(output, target):
    dc_list = []
    # 转换数据类型，方便进行计算
    output = output > 0.5
    target = target > 0.5
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    for i in range(output.shape[0]):
        output_i = output[i, :, :, :]
        target_i = target[i, :, :, :]
        output_i = np.atleast_1d(output_i.astype(np.bool_))
        target_i = np.atleast_1d(target_i.astype(np.bool_))

        intersection = np.count_nonzero(output_i & target_i)

        size_i1 = np.count_nonzero(output_i)
        size_i2 = np.count_nonzero(target_i)

        try:
            dc = 2. * intersection / (float(size_i1 + size_i2) + 1e-6)
        except ZeroDivisionError:
            dc = 0.0
        finally:
            dc_list.append(dc)
    dc = np.mean(dc_list)
    return dc


# Jaccard Coefficient
def jc(result, reference):
    """
    Jaccard coefficient

    Computes the Jaccard coefficient between the binary objects in two images.

    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.

    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    # 转换数据类型，方便进行计算
    result = result > 0.5
    reference = reference > 0.5
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    union = np.count_nonzero(result | reference)

    jc = float(intersection) / (float(union) + 1e-6)

    return jc


def recall(result, reference):
    """
    Recall.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.

    See also
    --------
    :func:`precision`

    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / (float(tp + fn) + 1e-6)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def ACC(result, reference):
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference)

    try:
        ACC = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    except ZeroDivisionError:
        ACC = 0.0

    return ACC


def ravd(result, reference):
    """
    Relative absolute volume difference.

    Compute the relative absolute volume difference between the (joined) binary objects
    in the two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    ravd : float
        The relative absolute volume difference between the object(s) in ``result``
        and the object(s) in ``reference``. This is a percentage value in the range
        :math:`[-1.0, +inf]` for which a :math:`0` denotes an ideal score.

    Raises
    ------
    RuntimeError
        If the reference object is empty.

    See also
    --------
    :func:`dc`
    :func:`precision`
    :func:`recall`

    Notes
    -----
    This is not a real metric, as it is directed. Negative values denote a smaller
    and positive values a larger volume than the reference.
    This implementation does not check, whether the two supplied arrays are of the same
    size.

    Examples
    --------
    Considering the following inputs

    >>> import numpy
    >>> arr1 = numpy.asarray([[0,1,0],[1,1,1],[0,1,0]])
    >>> arr1
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> arr2 = numpy.asarray([[0,1,0],[1,0,1],[0,1,0]])
    >>> arr2
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])

    comparing `arr1` to `arr2` we get

    >>> ravd(arr1, arr2)
    -0.2

    and reversing the inputs the directivness of the metric becomes evident

    >>> ravd(arr2, arr1)
    0.25

    It is important to keep in mind that a perfect score of `0` does not mean that the
    binary objects fit exactely, as only the volumes are compared:

    >>> arr1 = numpy.asarray([1,0,0])
    >>> arr2 = numpy.asarray([0,0,1])
    >>> ravd(arr1, arr2)
    0.0

    """
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    # if 0 == vol2:
    #     raise RuntimeError('The second supplied array does not contain any binary object.')

    return (vol1 - vol2) / float(vol2 + 1e-6)


def compute_metrics(output: torch.Tensor, label: torch.Tensor, out_channels: int, smooth: float = 1.0) -> Dict[
    str, Union[float, List[Dict[str, float]]]
]:
    """
    计算评估指标（IoU, Dice, Recall, ACC, RAVD）并记录，仅记录指标，不考虑类别不平衡。

    参数:
        output: 模型输出，单目标为 [B, 1, H, W]（logits），多目标为 [B, num_classes, H, W]（logits）
        label: 标签，单目标为 [B, 1, H, W]（0 或 1），多目标为 [B, H, W]（[0, num_classes-1]）
        num_classes: 类别数，单目标为 2，多目标为实际类别数（例如 ACDC 为 3）
        smooth: 平滑项，防止除零，默认 1.0

    返回:
        字典，包含平均指标（iou, dice, recall, acc, rvd）和每类指标列表（metrics_list）
    """
    # 输入校验
    batch_size, _, H, W = output.shape
    if out_channels == 1:
        if output.shape[1] != 1 or label.shape != (batch_size, 1, H, W):
            raise ValueError(
                f"二分类期望 output: [B, 1, H, W], label: [B, 1, H, W]，得到 output: {output.shape}, label: {label.shape}"
            )
        if label.max() > 1 or label.min() < 0:
            raise ValueError(f"二分类标签值应为 0 或 1，得到 min: {label.min()}, max: {label.max()}")
    else:
        if output.shape[1] != out_channels or label.shape != (batch_size, H, W):
            raise ValueError(
                f"多分类期望 output: [B, {out_channels}, H, W], label: [B, H, W]，得到 output: {output.shape}, label: {label.shape}"
            )
        if label.max() >= out_channels or label.min() < 0:
            raise ValueError(
                f"多分类标签值应为 [0, {out_channels-1}]，得到 min: {label.min()}, max: {label.max()}"
            )

    if out_channels == 1:
        # 二分类：前景类指标
        preds = (torch.sigmoid(output) > 0.5).float()  # [B, 1, H, W]
        metrics, sample_metrics = compute_class_metrics(preds, label, smooth)
        # metrics = compute_class_metrics(preds, label, smooth)
        return {
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "recall": metrics["recall"],
            "acc": metrics["acc"],
            "rvd": metrics["rvd"],
            "metrics_list": sample_metrics,  # 统一格式，记录前景类
        }
    else:
        # 多分类：每类指标
        preds = torch.softmax(output, dim=1).argmax(dim=1)  # [B, H, W]
        # mean_metrics, metrics_list = compute_multiclass_metrics(preds, label, out_channels, smooth)
        mean_metrics, sample_metrics = compute_multiclass_metrics(preds, label, out_channels, smooth)
        return {
            "iou": mean_metrics["iou"],
            "dice": mean_metrics["dice"],
            "recall": mean_metrics["recall"],
            "acc": mean_metrics["acc"],
            "rvd": mean_metrics["rvd"],
            "metrics_list": sample_metrics
        }


def compute_class_metrics(preds: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0) -> Dict[str, float]:
    """
    计算单类指标（IoU, Dice, Recall, ACC, RAVD），适用于二分类或多分类单类。

    参数:
        preds: 预测，二值化张量 [B, 1, H, W] 或 [B, H, W]
        labels: 标签，二值化张量 [B, 1, H, W] 或 [B, H, W]
        smooth: 平滑项，防止除零

    返回:
        指标字典，平均值（批次均值）
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)  # [B, H, W]
        labels = labels.squeeze(1)

    preds_flat = preds.view(preds.shape[0], -1)  # [B, H*W]
    labels_flat = labels.view(labels.shape[0], -1)

    intersection = (preds_flat * labels_flat).sum(dim=1)  # [B]
    pred_sum = preds_flat.sum(dim=1)  # [B]
    label_sum = labels_flat.sum(dim=1)  # [B]
    union = pred_sum + label_sum - intersection  # [B]

    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred_sum + label_sum + smooth)

    tp = intersection
    fn = label_sum - intersection
    tn = ((1 - preds_flat) * (1 - labels_flat)).sum(dim=1)
    fp = pred_sum - intersection

    recall = (tp + smooth) / (tp + fn + smooth)
    acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    rvd = (pred_sum - label_sum) / (label_sum + smooth)  # 相对体积差

    sample_metrics = [
        {
            "iou": iou[b].item(),
            "dice": dice[b].item(),
            "recall": recall[b].item(),
            "acc": acc[b].item(),
            "rvd": rvd[b].item()
        }
        for b in range(preds.shape[0])
    ]

    mean_metrics = {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "recall": recall.mean().item(),
        "acc": acc.mean().item(),
        "rvd": rvd.mean().item()
    }

    return mean_metrics, sample_metrics


def compute_multiclass_metrics(
    preds: torch.Tensor, labels: torch.Tensor, out_channels: int, smooth: float = 1.0
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    计算多分类指标（IoU, Dice, Recall, ACC, RAVD），记录每类指标。

    参数:
        preds: 预测类别索引 [B, H, W]
        labels: 标签类别索引 [B, H, W]
        num_classes: 类别数
        smooth: 平滑项，防止除零

    返回:
        平均指标字典（简单均值），每类指标列表
    """
    batch_size = preds.shape[0]
    sample_metrics = [[] for _ in range(batch_size)]
    metrics_sum = {"iou": 0.0, "dice": 0.0, "recall": 0.0, "acc": 0.0, "rvd": 0.0}
    # metrics_sum：所有类的一个batch的指标和
    for cls in range(1, out_channels):   # 从1开始表示不计算背景部分的指标
        pred_cls = (preds == cls).float()  # [B, H, W]
        label_cls = (labels == cls).float()

        # metrics = compute_class_metrics(pred_cls.unsqueeze(1), label_cls.unsqueeze(1), smooth)
        cls_metrics, cls_sample_metrics = compute_class_metrics(pred_cls.unsqueeze(1), label_cls.unsqueeze(1), smooth)
        # cls_metrics: 某一类的一个bacth的指标均值，cls_sample_metrics： 某一类的一个batch的每个样本的指标值
        for key in metrics_sum:
            metrics_sum[key] += cls_metrics[key]

        for b in range(batch_size):
            sample_metrics[b].append(cls_sample_metrics[b])  # sample_metrics：所有类的所有batch的每一个样本的指标值

    # 前景类平均（简单均值）
    num_foreground = max(1, out_channels - 1)
    mean_metrics = {key: value / num_foreground for key, value in metrics_sum.items()}
    # mean_metrics：所有类的一个batch的指标均值
    return mean_metrics, sample_metrics

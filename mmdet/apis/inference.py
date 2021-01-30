import mmcv
import numpy as np
import torch

from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform      # 图片转换
from mmdet.core import get_classes


def _prepare_data(img, img_transform, cfg, device):  # 数据预处理，cfg为配置文件
    ori_shape = img.shape         # 原始形状
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale=cfg.data.test.img_scale)
    img = to_tensor(img).to(device).unsqueeze(0)  # 图像转换为tensor
    img_meta = [
        # meta是指元素可提供相关页面的元信息
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

# 以单个下划线开头的变量或方法仅供内部使用，但是不强制执行。通过类名.变量名依然可以引用。
# 但是在使用通配符导入模块（from 模块 import *）时，不能调用使用下划线定义的函数，而常规导入（import 模块）是可以调用的。
def _inference_single(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():        
    # torch.no_grad()是一个上下文管理器,被该语句 wrap 起来的部分将不会track梯度。
    # with：方便需要事先设置，事后清理的工作
        result = model(return_loss=False, rescale=True, **data)  # module模型
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):    # inference推理；generator生成器
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)  # yield：生成器generator的一部分，有迭代功能的return


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)  # **用于参数前表示传入的(多个)参数将按照字典的形式存储，是一个字典
    model = model.to(device)   # 用指定的device设备训练
    model.eval()  # eval评估。和train相比不启用BatchNormalization和Dropout

    if not isinstance(imgs, list): # isinstance()：判断一个对象是否是一个已知的类型，类似type()，但会考虑子类继承父类
        return _inference_single(model, imgs, img_transform, cfg, device)  # 单个
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device) # 多个（生成器）


def show_result(img, result, dataset='coco', score_thr=0.3):
    class_names = get_classes(dataset)   # 获取类的名称
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)   # 构建被填充的数组 (shape形状为int元组几行几列, 填充的数)
        for i, bbox in enumerate(result)   # enumerate()将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，常用于 for 循环
    ]
    labels = np.concatenate(labels)   # 完成多个数组的拼接
    bboxes = np.vstack(result)      # 按垂直方向(行顺序)堆叠数组构成一个新的数组
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(        # 在一幅图上画出检测框
        img.copy(),             # 图片复制
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)

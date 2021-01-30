import mmcv
import numpy as np
import torch

from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform      #图片转换
from mmdet.core import get_classes


def _prepare_data(img, img_transform, cfg, device):  #数据预处理，cfg为配置文件
    ori_shape = img.shape         #原始形状
    img, img_shape, pad_shape, scale_factor = img_transform(
        img, scale=cfg.data.test.img_scale)
    img = to_tensor(img).to(device).unsqueeze(0)  #图像转换为tensor
    img_meta = [
        #meta是指元素可提供相关页面的元信息
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

#以单个下划线开头的变量或方法仅供内部使用，但是不强制执行。通过类名.变量名依然可以引用。
#但是在使用通配符导入模块（from 模块 import *）时，不能调用使用下划线定义的函数，而常规导入（import 模块）是可以调用的。
def _inference_single(model, img, img_transform, cfg, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():        
    #torch.no_grad()是一个上下文管理器,被该语句 wrap 起来的部分将不会track梯度。
    #with：方便需要事先设置，事后清理的工作
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):    #inference推理；generator生成器
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)  #yield：生成器generator的一部分，有迭代功能的return


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)   #用指定的device设备训练
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)


def show_result(img, result, dataset='coco', score_thr=0.3):
    class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)
    img = mmcv.imread(img)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)

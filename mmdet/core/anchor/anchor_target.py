import torch

from ..bbox import assign_and_sample, build_assigner, PseudoSampler, bbox2delta
from ..utils import multi_apply


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """
    
    Compute(计算) regression(回归) and classification(分类) targets(目标) for anchors.
    
    # 又名 get_targets   获得一个批次的训练和回归目标.
    
    Args:
        anchor_list:            (list[list[Tensor]])    所有批次所有尺度的 anchor 的列表,
                                                        每个 tensor 代表一张图片的一个尺度的 anchor, 形状为 (num_anchors, 4).
        valid_flag_list:        (list[list[Tensor]]):   所有批次所有尺度 anchor 的 valid flag,
                                                        每个 tensor 代表一张图片的一个尺度的 anchor 的 valid flag, 形状为 (num_anchors,).
        gt_bboxes_list:         (list[Tensor]):         一个 batch 的 gt bbox, 每个 tensor 的形状为 (num_gts, 4).
        img_metas:              (list[dict]):           一个 batch 的图片的属性信息.
        gt_bboxes_ignore_list:  (list[Tensor]):         需要忽略的 gt bboxes.
        gt_labels_list:         (list[Tensor] | None):  一个 batch 的 gt labels.
        label_channels:         (int):                  标签的通道.
        unmap_outputs:          (bool):                 是否填充 anchor 到没有筛选 valid flag 的长度.

    Returns:
            tuple:
                labels_list:        (list[Tensor]):     每个尺度的 label, 每个元素的形状为 (batch, n_anchors)
                label_weights_list: (list[Tensor]):     每个尺度 label 的权重, 每个元素的形状为 (batch, n_anchors)
                bbox_targets_list:  (list[Tensor]):     每个尺度的 bbox, 每个元素的形状为 (batch, n_anchors, 4)
                bbox_weights_list:  (list[Tensor]):     每个尺度 bbox 的权重, 每个元素的形状为 (batch, n_anchors, 4)
                num_total_pos:      (int):              一个批次所有图片的正样本总数
                num_total_neg:      (int):              一个批次所有图片的负样本总数
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
    # 将每张图片的 gt_bboxes都cat到一起, 每张图片的valid_flag_list也cat到一起
    # 对每一张图片调用 anchor_target_simple

    """
    
    
    num_imgs = len(img_metas)   # 计算 batch 的数量
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels   计算每个尺度 anchor 的数量 [187200, 46800, 11700, 2925, 780]
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    
    anchor_list = []         # 初始化 list
    valid_flag_list = []
    # concat(合并多个数组) all level anchors and flags to a single tensor  遍历每个图片, 合并每个图片中所有尺度的 anchor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])  # 检查长度是否相等
        # cat(concatenate) 拼接, 将两个tensor拼接在一起; (axis=0)将list中的元素(tensor)按行拼接
        anchor_list[i] = torch.cat(anchor_list[i])           # 合并所有尺度的 anchor  
        valid_flag_list[i] = torch.cat(valid_flag_list[i])   # 合并所有尺度的 flag
    
    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        # range() 创建一个整数列表, 一般用于 for 循环            
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]   # <class 'list'>: [None, None, None, None]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]              # <class 'list'>: [None, None, None, None]
        
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    
    # no valid anchors
    if any([labels is None for labels in all_labels]):   # any() 用于判断给定的可迭代参数 iterable 是否全为 False，若是则返回 False
        return None
      
    # sampled anchors of all images  统计所有 image 的正负样本
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])  # 正样本
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])  # 负样本
    
    # split(分开) targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
  
    """
    Convert targets by(of) image to targets by(of) feature level.   # 将图像中的目标(target) 转变为特征层中的目标

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    
    """
    
    target = torch.stack(target, 0)  # 沿着一个新维度对输入张量序列进行连接; 序列中所有的张量都应该为相同形状
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))    # append() 在列表末尾添加新对象;  squeeze(0) 压缩维度, 移除数组中维度为1的维度
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    """
    计算一张图片 anchor 的回归和分类的目标
    # 又名 _get_targets_single
    
    Args:
        flat_anchors:       (Tensor):   合并后的多尺度的 anchor. 形状为: (num_anchors ,4).
        valid_flags:        (Tensor):   合并后的多尺度的 anchor 的 flag, 形状为 (num_anchors,).
        gt_bboxes:          (Tensor):   图像的 ground truth bbox, 形状为 (num_gts, 4).
        gt_bboxes_ignore:   (Tensor):   需要忽略的 Ground truth bboxes 形状为: (num_ignored_gts, 4).
        img_meta:           (dict):     此图像的属性信息
        gt_labels:          (Tensor):   每个 box 的 Ground truth labels, 形状为 (num_gts,).
        label_channels:     (int):      label 所在的通道.
        unmap_outputs:      (bool):     是否将输出映射回原始 anchor 配置.

    Returns:
        tuple:
            labels:          (Tensor):    训练的标签, 形状为 (anchor 总数,)
            label_weights:   (Tensor):    训练标签的权重, 形状为 (anchor 总数,)
            bbox_targets:    (Tensor):    bbox 训练的目标值, 形状为 (anchor 总数, 4)
            bbox_weights:    (Tensor):    bbox 训练目标值的权重, 形状为 (anchor 总数, 4)
            pos_inds:        (Tensor):    正样本的索引, 形状为 (正样本总数,)
            neg_inds:        (Tensor):    负样本的索引, 形状为 (负样本总数,)
            
            
    1. 筛选出有效的 anchor
    2. anchor 分配正负样本(assigner) & anchor 正负样本采样(sampler)
    3. 构建 bbox 和 label 的目标和权重
      (1) 构建 bbox 的目标和权重：
        将正样本的 anchor 编码为中心点坐标，宽和高的偏移量。
        将正样本对应的 indices 设置为编码后的 anchor
        将正样本的权重设置为 1
      (2) 构建 label 的目标和权重：
        将正样本的 label 设置为 1，代表前景
        将正样本的权重设置为 1
    4. 填充 anchor 到没有筛选 valid flag 的长度
        
    """
   
# ========================= 1. 筛选出有效的 anchor ===========================
    # 获得有效的 flag, 这里的 inside_flags 就等于 valid_flags, 形状为 (num_anchors,)
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    
    # 如果 anchor 没有一个有效, 直接返回
    if not inside_flags.any():
        return (None, ) * 6
      
    # 筛选有效的 anchor, 此时 anchor 数量会减少为有效的 anchor 数量.
    anchors = flat_anchors[inside_flags, :]

# ============== 2. anchor 分配(assign)正负样本 & 正负样本采样(sample) ===============
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, None, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, None,
                                             gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        
# ======================== 3. 构建 label 和 bbox 的目标和权重 =====================

    num_valid_anchors = anchors.shape[0]      # 有效的 anchor 数量
    bbox_targets = torch.zeros_like(anchors)  # bbox 目标, 初始化将目标设置为 0
    bbox_weights = torch.zeros_like(anchors)  # bbox 权重, 即是否需要算入损失, 是否需要网络学习. 初始化将权重设置为 0
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)    # label 的目标, (初始化先将所有有效的 anchor 的标签标记为背景 (0))
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)   # label 的权重, 初始化将将权重权设置为 0

    pos_inds = sampling_result.pos_inds   # 获得正负样本的索引
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
    # ================ （1）构建 bbox 的目标和权重 ====================
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        
        # 将正样本对应的 indices 设置为编码后的 anchor, 将权重设置为 1
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        
    # ================ （2）构建 label 的目标和权重 ===================
        if gt_labels is None:      # 只有 rpn 的 gt_labels 才设置为 None
            labels[pos_inds] = 1
        else:                      # 否则设置为对应的类别编号
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        
        # 将正样本的权重设置为 1
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
        
# ===================== 4. 填充 anchor 到没有筛选 valid flag 的长度. ==================
    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)                  # 填充 labels
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)    # 填充 label_weights
        if label_channels > 1:
            labels, label_weights = expand_binary_labels(
                labels, label_weights, label_channels)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)      # 填充 bbox_targets
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)      # 填充 bbox_weights

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights



def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \                    # \(末尾) 续行符
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ 
    Unmap a subset of item (data) back to the original set of items (of size count) 
    # 将项目(数据)的一个子集映射回原始的项目集(大小计数)
    """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

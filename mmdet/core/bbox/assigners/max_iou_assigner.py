import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt_bbox or background to each bbox.  为每个bbox分配一个相应的gt_bbox或背景

    Each proposals will be assigned with `-1`, `0`, or a positive integer indicating the ground truth index.  每个提案将被分配' -1 '，' 0 '，或一个表示ground truth指数的正整数

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox            这个方法将gt bbox分配给每个bbox(提案/锚点)，每个bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,            将被赋值为-1、0或一个正数。-1表示不关注，
        0 means negative sample, positive number is the index (1-based) of assigned gt.    0表示负样本，正数为分配 gt 的 index (1-based)
        The assignment is done in following steps, the order matters.                      分配按以下步骤进行，顺序很重要。
        
        1. assign every bbox to -1                                              # 初始化时假设每个anchor的mask都是-1，表示都是忽略anchor
        2. assign proposals whose iou with all gts < neg_iou_thr to 0           # 将每个anchor和所有gt的iou的最大Iou小于neg_iou_thr的anchor的mask设置为0，表示是负样本(背景样本)
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,        # 对于每个anchor，计算其和所有gt的iou，选取最大的iou对应的gt位置，如果其最大iou大于等于pos_iou_thr，
           assign it to that bbox                                                  则设置该anchor的mask设置为1，表示该anchor负责预测该gt bbox,是高质量anchor
           
        # 3的设置可能会出现某些gt没有分配到对应的anchor(由于iou低于pos_iou_thr)，故下一步对于每个gt还需要找出和最大iou的anchor位置，
          如果其iou大于min_pos_iou，将该anchor的mask设置为1，表示该anchor负责预测对应的gt。
          通过本步骤，可以最大程度保证每个gt都有anchor负责预测，如果还是小于min_pos_iou，那就没办法了，只能当做忽略样本了。
          从这一步可以看出，3和4有部分anchor重复分配了，即当某个gt和anchor的最大iou大于等于pos_iou_thr，那肯定大于min_pos_iou，此时3和4步骤分配的同一个anchor。
        4. for each gt bbox, assign its nearest proposals (may be more than one) to itself    # 对于每个gt bbox，将其最近的提案(可以不止一个)分配给自己

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
       
       """
        
        
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]                            # bboxes   对2000个proposals, 为 tensor(2000,4)
        overlaps = bbox_overlaps(gt_bboxes, bboxes)       # 假设有10个gt, 则做iou后, overlaps 为 tensor(10, 2000)

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            ignore_overlaps = bbox_overlaps(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_bboxes_inds = torch.nonzero(
                ignore_max_overlaps > self.ignore_iof_thr).squeeze()
            if ignore_bboxes_inds.numel() > 0:
                overlaps[ignore_bboxes_inds[:, 0], :] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)     # 开始分配正负例
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        
        
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)   # num_gts 真实labels个数

        # 1. assign -1 by default   所有 index(指标, 指数) 设置为 -1, 表示被忽略的 anchor 
        assigned_gt_inds = overlaps.new_full(
            (num_bboxes, ), -1, dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts        计算每个 anchor, 和哪个 gt 的 iou 最大
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals      计算每个 gt, 和哪个 anchor 的 iou 最大, 可能两个 max 的索引有重复
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below     对于每个anchor，如果其和gt的最大iou都小于neg_iou_thr阈值，则分配负样本
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold      对于每个anchor，如果其和gt的最大iou大于pos_iou_thr阈值，则分配正样本
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU    对于每个gt, 如果其和某个anchor的最大iou大于min_pos_iou阈值，那么依然分配正样本
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                # 该参数的含义是: 当某个gt, 和其中好几个anchor都是最大iou(最大iou对应的anchor有好几个的时候)，则全部分配正样本
                # 该操作可能会出现某几个anchor和同一个Gt匹配，都负责预测
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    # 仅仅考虑最大的一个, 不考虑多个最大时候
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

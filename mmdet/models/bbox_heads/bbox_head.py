import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)
from ..registry import HEADS


@HEADS.register_module
class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively   最简单的 RoI head, 只有两个全连接层, 分别为分类和回归
    
    
    """

    def __init__(self,
                 with_avg_pool=False,         # 平均池化（下采样）
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic      # 回归_分类_不可知

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)     # 下采样
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)    # c *= a 等效于 c = c * a
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)    # 用于分类的全连接层
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)      # 用于回归的全连接层
        self.debug_imgs = None

    def init_weights(self):    # 初始化权重, 使用 torch.nn.init 初始化函数
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)   # 正态分布 - N(mean, std)
            nn.init.constant_(self.fc_cls.bias, 0)         # 常数 - 固定值
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)    # 展平处理, 因为接下来要通过全连接层
        cls_score = self.fc_cls(x) if self.with_cls else None    # 通过分类的全连接层得到 cls_score
        bbox_pred = self.fc_reg(x) if self.with_reg else None    # 通过回归的全连接层得到 bbox_pred
        
        return cls_score, bbox_pred

      
      
    def get_target(self, 
                   sampling_results, 
                   gt_bboxes, 
                   gt_labels,
                   rcnn_train_cfg):
      
    """Calculate the ground truth for all samples in a batch according to
        the sampling_results.
        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single`(见最新版本的 mmdetection/mmdet/models/roi_heads/bbox_heads/bbox_head.py) function.
       
       Args:
            sampling_results (List[obj:SamplingResults]): Assign(分配) results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:
                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) 
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) 
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) 
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) 
        """
    
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True):
        losses = dict()          # loss 为一个字典，存储分类和回归的损失，以及准确率
        if cls_score is not None:
            losses['loss_cls'] = weighted_cross_entropy(
                cls_score, labels, label_weights, reduce=reduce)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            losses['loss_reg'] = weighted_smoothl1(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=bbox_targets.size(0))
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
      
        """Transform network output for a batch into bbox predictions.
        If the input rois has batch dimension, the function would be in
        `batch_mode` and return is a tuple[list[Tensor], list[Tensor]],
        otherwise, the return is a tuple[Tensor, Tensor].
        
        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5)
               or (B, num_boxes, 5)
            cls_score (list[Tensor] or Tensor): Box scores for
               each scale level, each is a 4D-tensor, the channel number is
               num_points * num_classes.
            bbox_pred (Tensor, optional): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_classes * 4.
            img_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]], optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If rois shape is (B, num_boxes, 4), then
                the max_shape should be a Sequence[Sequence[int]]
                and the length of max_shape should also be B.
            scale_factor (tuple[ndarray] or ndarray): Scale factor of the
               image arange as (w_scale, h_scale, w_scale, h_scale). In
               `batch_mode`, the scale_factor shape is tuple[ndarray].
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None
            
        Returns:
            tuple[list[Tensor], list[Tensor]] or tuple[Tensor, Tensor]:
            
                If the input has a batch dimension, the return value is
                a tuple of the list. The first list contains the boxes of
                the corresponding image in a batch, each tensor has the
                shape (num_boxes, 5) and last dimension 5 represent
                (tl_x, tl_y, br_x, br_y, score). Each Tensor in the second
                list is the labels with shape (num_boxes, ). The length of
                both lists should be equal to batch_size. Otherwise return
                value is a tuple of two tensors, the first tensor is the
                boxes with scores, the second tensor is the labels, both
                have the same shape as the first case.
        """
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]
            # TODO: add clip here

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)

            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.   改善bbox

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

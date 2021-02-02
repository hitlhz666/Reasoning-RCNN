from .geometry import bbox_overlaps
from .assigners import BaseAssigner, MaxIoUAssigner, AssignResult
from .samplers import (BaseSampler, PseudoSampler, RandomSampler,
                       InstanceBalancedPosSampler, IoUBalancedNegSampler,
                       CombinedSampler, SamplingResult, RandomSamplerFixnum)
from .assign_sampling import build_assigner, build_sampler, assign_and_sample
from .transforms import (bbox2delta, delta2bbox, bbox_flip, bbox_mapping,
                         bbox_mapping_back, bbox2roi, roi2bbox, bbox2result)
from .bbox_target import bbox_target


"""

assign和sample是在anchor target中的核心操作
assign一般基于IOU, mmdet中也有基于atss和基于point的等
sample一般为随机, 也有ohem的, 基于伪标签的
"""



__all__ = [
    'bbox_overlaps', 'BaseAssigner', 'MaxIoUAssigner', 'AssignResult',
    'BaseSampler', 'PseudoSampler', 'RandomSampler',
    'InstanceBalancedPosSampler', 'IoUBalancedNegSampler', 'CombinedSampler',
    'SamplingResult', 'build_assigner', 'build_sampler', 'assign_and_sample',
    'bbox2delta', 'delta2bbox', 'bbox_flip', 'bbox_mapping',
    'bbox_mapping_back', 'bbox2roi', 'roi2bbox', 'bbox2result', 'bbox_target',
    'RandomSamplerFixnum'
]

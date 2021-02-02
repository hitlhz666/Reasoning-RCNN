from .base_assigner import BaseAssigner
from .max_iou_assigner import MaxIoUAssigner
from .assign_result import AssignResult

"""

assign和sample是在anchor target中的核心操作
assign一般基于IOU, mmdet中也有基于atss和基于point的等
sample一般为随机, 也有ohem的, 基于伪标签的
"""

__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult']

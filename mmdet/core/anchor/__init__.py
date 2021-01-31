# 在pytorch中， AnchorGenerator主要用于生成候选框，该类存储在torchvision/models/detection/rpn.py中
# 用generate_anchors产生多种坐标变换，这种坐标变换由scale和ratio来，相当于提前计算好。
# anchor_target_layer先计算的是从feature map映射到原图的中点坐标，然后根据多种坐标变换生成不同的框。

anchor_target_layer层是产生在rpn训练阶段产生anchors的层

from .anchor_generator import AnchorGenerator
from .anchor_target import anchor_target

__all__ = ['AnchorGenerator', 'anchor_target']

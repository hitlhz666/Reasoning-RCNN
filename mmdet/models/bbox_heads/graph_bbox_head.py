import torch.nn as nn
import torch
from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
import torch.nn.functional as F
from mmdet.core import (weighted_cross_entropy, weighted_smoothl1, accuracy)

# 本模块为加入推理型知识图谱

@HEADS.register_module          # mmdetection中注册模块
class GraphBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches. 共享卷积层和全连接层

                                /-> cls convs -> cls fcs -> cls  
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605    no quality assurance 无质量保证

    def __init__(self,
                 num_attr_conv=0,
                 num_rela_conv=0,
                 num_spat_conv=0,
                 with_attr=False,             # 添加属性信息
                 with_rela=False,             # 添加关系信息
                 with_spat=False,
                 num_spat_graph=10,
                 graph_out_channels=256,
                 nf=64,
                 ratio=[4, 2, 1],             # 比率
                 normalize=None,
                 num_shared_fcs=0,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        
        super(GraphBBoxHead, self).__init__(*args, **kwargs)
        # original FPN head   原始的 FPN head
        self.num_shared_fcs = num_shared_fcs      # 全连接层数
        self.normalize = normalize
        self.with_bias = normalize is None
        self.fc_out_channels = fc_out_channels
        
        
        # add shared convs and fcs      添加共享的卷积层和全连接层
        _, self.shared_fcs, last_layer_dim = \              # 表示接着下一行
            self._add_conv_fc_branch(0, self.in_channels, num_branch_fcs=self.num_shared_fcs)
        
        
        if num_shared_fcs > 0:
            self.cls_last_dim = last_layer_dim
            self.reg_last_dim = last_layer_dim
            self.in_channels = last_layer_dim
        else:
            self.cls_last_dim = self.in_channels
            self.reg_last_dim = self.in_channels

            
        # corresponding to graph compute  相应的图谱计算
        self.attr_transferW = nn.ModuleList()
        self.rela_transferW = nn.ModuleList()
        self.spat_transferW = nn.ModuleList()
        if with_attr:                # 如果有属性知识
            self.attr_convs, _, _ = self._add_conv_fc_branch(num_attr_conv, self.in_channels, nf, ratio)
            self.attr_transferW = nn.Linear(self.in_channels, graph_out_channels)    # 全连接层
            self.cls_last_dim = self.cls_last_dim + graph_out_channels
            self.reg_last_dim = self.reg_last_dim + graph_out_channels
        if with_rela:                # 如果有关系知识
            self.rela_convs, _, _ = self._add_conv_fc_branch(num_rela_conv, self.in_channels, nf, ratio)
            self.rela_transferW = nn.Linear(self.in_channels, graph_out_channels)
            self.cls_last_dim = self.cls_last_dim + graph_out_channels
            self.reg_last_dim = self.reg_last_dim + graph_out_channels
        if with_spat:
            self.spat_convs, _, _ = self._add_conv_fc_branch(num_spat_conv, 5, nf=5, ratio=[1])
            self.spat_transferW = nn.Linear(self.in_channels, graph_out_channels)
            self.cls_last_dim = self.cls_last_dim + graph_out_channels
            self.reg_last_dim = self.reg_last_dim + graph_out_channels
        self.with_attr = with_attr
        self.with_rela = with_rela
        self.with_spat = with_spat
        self.num_spat_graph = num_spat_graph

        
        # classifer and bbox regression 分类和边界框回归
        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed  因为输入通道已更改，所以重建fc_cls和fc_reg（全连接层对应的类别和回归）
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else
                           4 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)


            
    # add shared convs and fcs     函数作用：添加共享的卷积层和全连接层
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            in_channels,
                            nf=0,
                            ratio=[0],
                            num_branch_fcs=0):
        """Add shared or separable branch   添加共享或单独的分支

        convs -> avg pool (optional) -> fcs        卷积层 -> 平均池化层 (可选) -> 全连接层
        """
        last_layer_dim = in_channels
        # add branch specific conv layers 添加特定的卷积层分支
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            assert num_branch_convs == len(ratio) + 1
            for i in range(num_branch_convs):
                conv_in_channels = (last_layer_dim
                                    if i == 0 else conv_out_channels)
                conv_out_channels = (int(nf * ratio[i])
                                     if i < num_branch_convs - 1 else 1)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        conv_out_channels,
                        1,
                        normalize=self.normalize,
                        bias=self.with_bias))

        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim
                                  if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels

        return branch_convs, branch_fcs, last_layer_dim


    def init_weights(self):  # 初始化权重
        super(GraphBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.attr_transferW, self.rela_transferW, self.spat_transferW]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, geom_f, bs):
        # shared part
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        if x.dim() > 2:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)   # 展平处理
        feat_dim = x.size(1)
        x = x.view(bs, -1, feat_dim)

        # compute A adj matrix     计算A矩阵（A矩阵表示加入的知识图谱信息）
        a_super = []
        enhanced_feat = []
        if self.with_attr or self.with_rela:
            # 当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，
            # 这时候我们就需要使用detach()函数来切断一些分支的反向传播
            W1 = x.detach().unsqueeze(2)     # pytorch.detach().detach_() 返回一个新的从当前图中分离的Variable，返回的 Variable 不会梯度更新
            W2 = torch.transpose(W1, 1, 2)   # torch.transpose 交换 tensor 的两个维度
            diff_W = torch.abs(W1 - W2)
            diff_W = torch.transpose(diff_W, 1, 3)
            
            
            if self.with_attr:
                A_a = diff_W
                for conv in self.attr_convs:
                    A_a = conv(A_a)
                A_a = A_a.contiguous()  # 将 tensor 在内存中的位置变为连续的，方便接下来 view() / squeeze() 
                A_a = A_a.squeeze(1)
                a_super.append(A_a)  # 增加到列表中
                # propogation  建议
                enhanced_feat.append(self.propagate_em(x, A_a, self.attr_transferW))

            if self.with_rela:
                A_r = diff_W
                for conv in self.rela_convs:
                    A_r = conv(A_r)
                A_r = A_r.contiguous()
                A_r = A_r.squeeze(1)
                a_super.append(A_r)
                # propogation
                enhanced_feat.append(self.propagate_em(x, A_r, self.rela_transferW))

                
        if self.with_spat:
            W1 = geom_f.unsqueeze(2)
            W2 = torch.transpose(W1, 1, 2)
            diff_W = W1 - W2
            diff_W = torch.transpose(diff_W, 1, 3)
            Iden = torch.eye(diff_W.size(-1)).cuda()
            A_s = W2.new_zeros((diff_W.size(-1), diff_W.size(-1)))
            for i in range(self.num_spat_graph):
                tmp_A = diff_W
                for conv in self.spat_convs:
                    tmp_A = conv(tmp_A)
                A_s = tmp_A + A_s + Iden
            A_s = A_s.contiguous()
            A_s = A_s.squeeze(1)
            enhanced_feat.append(self.propagate_em(x, A_s, self.spat_transferW))

        enhanced_feat = torch.cat(enhanced_feat, -1)    # torch.cat(_, -1)  在最后一个维度上拼接
        
        
        
        # separate branches
        assert len(x.size()) == len(enhanced_feat.size())     # 保证特征增强前后尺寸相同
        x = torch.cat((x, enhanced_feat), -1)
        x_cls = x.view(-1, x.size(-1))
        x_reg = x.view(-1, x.size(-1))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred, a_super

    
    
    def loss(self, cls_score, bbox_pred, A_pred, A_gt, labels, label_weights, bbox_targets,
             bbox_weights, reduce=True):
        losses = dict()
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
        if A_pred:
            assert len(A_pred) == len(A_gt)
            assert A_pred[0].size() == A_gt[0].size()
            num_a_pred = len(A_pred)
            for i in range(num_a_pred):
                losses['loss_adj' + str(i)] = F.mse_loss(A_pred[i], A_gt[i].detach())
        return losses

    def propagate_em(self, x, A, W):
        A = F.softmax(A, 2)
        x = torch.bmm(A, x)     # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b 的size为(b,w,h), 注意两个tensor的维度必须为3
        x = W(x)
        return x


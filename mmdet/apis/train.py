from __future__ import division   # python2.x中导入python未来支持的语言特征division(精确除法)

from collections import OrderedDict  # 用于获取有序字典

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
# parallel 并行
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel   # 重新封装了torch内部的并行计算，包括数据的collect、distribute、Scatter等，与cuda相关

from mmdet.core import (DistOptimizerHook, DistEvalmAPHook,
                        CocoDistEvalRecallHook, CocoDistEvalmAPHook)
from mmdet.datasets import build_dataloader
from mmdet.models import RPN
from .env import get_root_logger


def parse_losses(losses):   # 解析_损失
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():    # dict.items()返回可遍历的(键, 值)元组数组
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()   # 计算样本的平均值
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)  #求和

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()  #将单元素张量变为张量

    return loss, log_vars


def batch_processor(model, data, train_mode):   # 批_处理器
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,                    # 训练检测器
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)      # 获取日志信息

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)    # 分布式训练
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)   # 非分布式训练


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders 加载数据
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    
    # build runner 用来为pytorch操控安排训练过程中的各个环节的类，该类在mmcv/mmcv/runner/runner.py中
    # 这个操控包括，要在module中获取中间变量啊，或者加载和保存检查点，或者启动训练、启动测试、或者初始化权重。
    # 本身这个函数是不能改变这个网络模型的各个部分的，也就是说，我们要真正修改backbone、FPN、optimizer等，或者分类回归的具体实现，跟这个类无关。
    # 也就是说，你只要把你定义好的网络模型结构，加载好的数据集，你要的优化器等，扔给runner，它就会来帮你跑模型。
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
   
    # Optimizer优化器 是用来更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值，从而最小化(或最大化)损失函数E(x)
    # 这种算法使用各参数的梯度值来最小化或最大化损失函数E(x)。最常用的一阶优化算法是梯度下降。
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
   
    """
    # fp16 setting   用来提速的
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    """
    
    # register hooks 用于查看中间变量
    # hook（钩子）的作用是，当反传时，除了完成原有的反传，额外多完成一些任务。你可以定义一个中间变量的hook，将它的grad值打印出来，
    # 当然你也可以定义一个全局列表，将每次的grad值添加到里面去。下面的hooks也是一样的，具体见pytorch中hooks的作用
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    
    # 多GPU训练比单GPU训练多的代码
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:      # 如果验证成功
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(CocoDistEvalRecallHook(cfg.data.val))
        else:
            if cfg.data.val.type == 'CocoDataset':
                runner.register_hook(CocoDistEvalmAPHook(cfg.data.val))
            else:
                runner.register_hook(DistEvalmAPHook(cfg.data.val))

    # 选择从断点(resume)继续训练还是从头训练(load)
    if cfg.resume_from:   
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)  # workflow工作流


def _non_dist_train(model, dataset, cfg, validate=False):    # 单GPU训练
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

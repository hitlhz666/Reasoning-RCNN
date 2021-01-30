import logging   # 日志模块
import os      # 操作系统模块；文件/目录方法
import random   # 返回随机生成的一个实数

import numpy as np  # 数组
import torch  
import torch.distributed as dist  # 分布式训练，多GPU并行训练
import torch.multiprocessing as mp   # 用于在相同数据的不同进程中共享视图
from mmcv.runner import get_dist_info


def init_dist(launcher, backend='nccl', **kwargs):      # *args传递一个可变参数列表给函数实参；**kwargs将一个可变关键字参数的字典传给函数实参
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')            # 设置进程启动方式为spawn
    if launcher == 'pytorch':                     # launcher启动器
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()         # gpu个数
    torch.cuda.set_device(rank % num_gpus)    # 设置当前设备
    dist.init_process_group(backend=backend, **kwargs)   # 使用 init_process_group 设置GPU 之间通信使用的后端和端口


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError              # 显示错误


def _init_dist_slurm(backend, **kwargs):
    raise NotImplementedError


def set_random_seed(seed): 
    # 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    # 使得每次运行该 .py 文件时生成的随机数相同。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()      # 初始化
    if not logger.hasHandlers():  # 判断是否有处理器
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()      # 获得分布式训练的信息
    if rank != 0:
        logger.setLevel('ERROR')
    return logger

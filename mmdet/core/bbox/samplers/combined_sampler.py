from .base_sampler import BaseSampler
from ..assign_sampling import build_sampler


class CombinedSampler(BaseSampler):

    def __init__(self, pos_sampler, neg_sampler, **kwargs):
        super(CombinedSampler, self).__init__(**kwargs)               # super(父类,self).__init__() 可以指定调用的父类
        self.pos_sampler = build_sampler(pos_sampler, **kwargs)       # **kwargs用作传递键值可变长参数列表(比如字典)
        self.neg_sampler = build_sampler(neg_sampler, **kwargs)

    def _sample_pos(self, **kwargs):
        raise NotImplementedError              # 子类没有实现父类要求一定要实现的接口

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

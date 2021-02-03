import numpy as np
import torch

from .random_sampler import RandomSampler

"""
字面意思就是每个instance抽样出来的样本要均衡，不能cat抽个1000个，dog只抽个10个
改写了RandomSampler里的sample_pos函数，同样也是改了最后那个pos_index大于expected的那部分

"""
class InstanceBalancedPosSampler(RandomSampler):

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)      # 首先看一下给的bboxes里面有哪些label是大于0的, 得到了他们的index
        
        if pos_inds.numel() != 0:                   # 首先只要这个index的数目不是0个, 这些就都可以是positive sample
            pos_inds = pos_inds.squeeze(1)
        
        if pos_inds.numel() <= num_expected:        # 当pos_indxs的数目小于想要的sample的数目的时候, 就直接用这个pos_index
            return pos_inds
        else:                                       # 反之就从这么多index里 用某种方法(区别在这里: 随机/类均衡法/OHEM法) 采样num_expected个出来
            unique_gt_inds = assign_result.gt_inds[pos_inds].unique()
            num_gts = len(unique_gt_inds)
            num_per_gt = int(round(num_expected / float(num_gts)) + 1)
            sampled_inds = []
            for i in unique_gt_inds:
                inds = torch.nonzero(assign_result.gt_inds == i.item())
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue
                if len(inds) > num_per_gt:
                    inds = self.random_choice(inds, num_per_gt)
                sampled_inds.append(inds)
            sampled_inds = torch.cat(sampled_inds)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                extra_inds = torch.from_numpy(extra_inds).to(
                    assign_result.gt_inds.device).long()
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            elif len(sampled_inds) > num_expected:
                sampled_inds = self.random_choice(sampled_inds, num_expected)
            return sampled_inds

import torch


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)  # ratios 比率(复数)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    # @property 装饰器会将方法转换为相同名称的只读属性, 可以与所定义的属性配合使用, 这样可以防止属性被修改
    @property     # 装饰器, 创建只读属性
    def num_base_anchors(self):
        return self.base_anchors.size(0)   # size(a,0)返回该二维矩阵的行数

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
            
            
        h_ratios = torch.sqrt(self.ratios)     # 逐元素计算张量的平方根
        w_ratios = 1 / h_ratios
        if self.scale_major:
            # view()返回的数据和传入的tensor一样，只是形状不同          
            # -1本意是根据另外一个数来自动调整维度，但是这里只有一个维度, 因此就会将X里面的所有维度数据转化成一维的, 并且按先后顺序排列
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1) 
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # stack()拼接函数 沿着一个新维度对输入张量序列进行连接, 序列中所有的张量都应该为相同形状
        # 把多个2维的张量凑成一个3维的张量; 多个3维的凑成一个4维的张量...即在增加新的维度上进行堆叠
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()  # dim=-1 直接按照逐个元素的累加方法进行; round() 返回浮点数x的四舍五入值

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))     # 重复单个数
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)   # view(-1, 1) 得到一个列tensor
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):   # 网格_锚点; Stride = 每像素占用的字节数(像素位数/8) * Width(每行的像素个数)
        base_anchors = self.base_anchors.to(device)    # to(device)在指定设备上训练

        feat_h, feat_w = featmap_size                  # 从特征图的尺寸中获取高和宽
        shift_x = torch.arange(0, feat_w, device=device) * stride  # arange() 返回一维大小的张量 tensor([0, 1, 2, ..., feat_w-1])
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid

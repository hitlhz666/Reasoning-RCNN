import torch


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size       # 基础长宽
        self.scales = torch.Tensor(scales)  # 放大倍数
        self.ratios = torch.Tensor(ratios)  # 长宽比
        self.scale_major = scale_major    # 排列方式, 前m个代表m种scale还是m种ratio
        self.ctr = ctr   # centre 中心
        
        # base_anchor有三个, 这时因为有三个ratio, 一个scale, 中心在 base_size/2 处, 以左上右下表示
        # 生成方式在 self.gen_base_anchors 中
        self.base_anchors = self.gen_base_anchors()

    # @property 装饰器会将方法转换为相同名称的只读属性, 可以与所定义的属性配合使用, 这样可以防止属性被修改
    @property     # 装饰器, 创建只读属性
    def num_base_anchors(self):
        return self.base_anchors.size(0)   # size(a,0)返回该二维矩阵的行数

    # base_size 由 stride赋值, 如若 anchor 的大小是 base_size/2 的话, 那么这些anchor构成了不相交的划分
    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr
            
        # ratio指的是高/宽  取一个根号   
        h_ratios = torch.sqrt(self.ratios)     # sqrt 逐元素计算张量的平方根
        w_ratios = 1 / h_ratios
        
        # 如果是scale_major，那么ratios变为3*1的矩阵，scale变为1 * 1（scale个数）的矩阵。最后相乘，以基础长宽为基础，最后拉长，得到w_s
        # 如果 scale 的个数为2，那么ws的前2个为ratios[0]乘上两个scales的w，之后两个为ratios[1]乘上两个scale的w
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
        
        # 最终返回左上角和右下角, 并取整
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):      # 生成网格点坐标
        xx = x.repeat(len(y))     # 重复单个数
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)   # view(-1, 1) 得到一个列tensor
        if row_major:
            return xx, yy
        else:
            return yy, xx
        
    # Stride(步长) = 每像素占用的字节数(像素位数/8) * Width(每行的像素个数)
    #  输入: 特征图尺寸, Stride, 生成meshgrid，以(0, 0)为左上角起点生成grid的(x, y)坐标，距离为stride   shift_xx,shift_yy
    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)    # to(device)在指定设备上训练

        feat_h, feat_w = featmap_size                  # 从特征图的尺寸中获取高和宽
        shift_x = torch.arange(0, feat_w, device=device) * stride  # arange() 返回一维大小的张量 tensor([0, 1, 2, ..., feat_w-1])
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)  #张量逐元素对应相加
        shifts = shifts.type_as(base_anchors)    # tensor1.type_as(tensor2) 将1的数据类型转换为2的数据类型
        # first feat_w elements(基本要素) correspond(相当于) to the first row of shifts(转移，变动)
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shifted anchors (K, A, 4), reshape(重塑形状) to (K*A, 4)
        
        # None是python中的一个特殊的常量，表示一个空的对象。
        # 数据为空并不代表是空对象，例如[],''等都不是None。
        # None有自己的数据类型NontType，你可以将None赋值给任意对象，但是不能创建一个NoneType对象。
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)  # -1表示不知道是 几*4，让系统自己判断
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        
        # grid_anchors获得的 anchor按照 level从低到高append起来, 得到anchor_list
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):    # valid 有效的; flag 旗
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w  # 断言, 用于报错
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)   # 零张量
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy    # & 位运算, 与运算
        valid = valid[:, None].expand(        # list.extend(list1) 参数必须是列表类型, 可以将参数中的列表合并到原列表的末尾
            valid.size(0), self.num_base_anchors).contiguous().view(-1)    # size(axis) axis = 0, 返回该二维矩阵的行数; axis = 1, 返回该二维矩阵的列数
            # contiguous() 把tensor变成在内存中连续分布的形式
        return valid

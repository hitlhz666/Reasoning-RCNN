from abc import ABCMeta, abstractmethod

# 多态: 一种事物的多种形态      例如 动物(animal): 猫  狗  .......
# 多态性是指具有不同功能的函数可以使用相同的函数名，这样就可以用一个函数名调用不同内容的函数。
# 在面向对象方法中一般是这样表述多态性：向不同的对象发送同一条消息，不同的对象在接收时会产生不同的行为（即方法）。
# 也就是说，每个对象可以用自己的方式去响应共同的消息。所谓消息，就是调用函数，不同的行为就是指不同的实现，即执行不同的函数。

class BaseAssigner(metaclass=ABCMeta):     # 同一类事物: BaseAssigner

    @abstractmethod       # 上述代码子类是约定俗称的实现这个方法, 加上 @abc.abstractmethod 装饰器后, 则严格控制子类必须实现这个方法
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass              # pass 是空语句, 是为了保持程序结构的完整性;     pass 不做任何事情, 一般用做占位语句

__version__ = '0.0.1'

from torch.optim.lr_scheduler import _LRScheduler

class LinearLR(_LRScheduler):
    r"""Set the learning rate of each parameter group with a linear
    schedule: :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0` is the
    initial lr, :math:`t` is the current epoch or iteration (zero-based) and
    :math:`T` is the total training epochs or iterations. It is recommended to
    use the iteration based calculation if the total number of epochs is small.
    When last_epoch=-1, sets initial lr as lr.
    It is studied in
    `Budgeted Training: Rethinking Deep Neural Network Training Under Resource
     Constraints`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Total number of training epochs or iterations.
        last_epoch (int): The index of last epoch or iteration. Default: -1.
        
    .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
    Resource Constraints:
        https://arxiv.org/abs/1905.04753
    """

    def __init__(self, optimizer, T, last_epoch=-1):
        self.T = float(T)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        rate = 1 - self.last_epoch/self.T
        return [rate*base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()

    
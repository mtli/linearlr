import torch
from linearlr import LinearLR
from torch import optim

# it is recommended to update the lr per iteration
# rather than per epoch, but both will work
max_iter = 100
base_lr = 1
dummy_optimizer = optim.SGD([torch.tensor(0)], base_lr)

scheduler = LinearLR(dummy_optimizer, max_iter)
for i in range(max_iter):
    # if pytorch < 1.1, update lr before training
    # scheduler.step()

    # do some training
    print('Iter %3d: lr: %g' % (i, scheduler.get_lr()[0]))

    # update scheduler (pytorch >= 1.1)
    scheduler.step()

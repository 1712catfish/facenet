import random
from functools import reduce

import torch


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


carrierMonad = lambda f: lambda x, y: (f(x), y)
lift = lambda e: lambda f: lambda *args: f(*(e(_) for _ in args))
pipe = lambda *fs: lambda *x: reduce(lambda v, f: f(*v) if isiterable(v) else f(v), fs, x)

tensorLift = lift(lambda x: torch.autograd.Variable(x))
cudaLift = lift(lambda x: x.cuda())
cudaTensorLift = lift(lambda x: torch.autograd.Variable(x).cuda())

cuda = lambda x: torch.autograd.Variable(x).cuda()

cpu = lambda x: x.cpu().detach().numpy()


def toss(probability):
    return random.random() < probability

import torch
from torch import nn

from sklearn.metrics import accuracy_score

from loader import build_loader
from loss import CleanContrastiveLoss
from model import Model, ArcMarginModel
from functions import tensor


def next_(train_iter, train_loader):
    try:
        inp, label = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        inp, label = next(train_iter)

    return tensor(inp), tensor(label), train_iter




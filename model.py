import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import config

from functions import tensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=False, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.classifier[1].out_features, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, config.EMB_SIZE),
        )

    def forward(self, inp, label):
        x = self.backbone(inp)
        x = self.head(x)
        return x, label


class ArcMarginModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(config.NUM_CLASSES, config.EMB_SIZE))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.m = 0.5
        self.s = 64.0

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, inp, label):
        x = F.normalize(inp)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size()).cuda()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output, label


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



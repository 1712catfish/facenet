import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanContrastiveLoss(nn.Module):
    def forward(self, output1, output2, label1, label2):
        y = torch.eq(label1, label2)
        y_hat = F.pairwise_distance(output1, output2, keepdim=True)
        return torch.mean(
            y * torch.pow(y_hat, 2) +
            (1 - y) * torch.pow(F.relu(1 - y_hat), 2)
        )

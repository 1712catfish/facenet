import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanContrastiveLoss(nn.Module):
    def forward(self, output1, output2, y):
        y_hat = F.pairwise_distance(output1, output2, keepdim=True)
        return torch.mean(
            y * torch.pow(y_hat, 2) +
            ~y * torch.pow(F.relu(2.0 - y_hat), 2)
        )

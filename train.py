import torch
from torch import nn

from sklearn.metrics import accuracy_score

from loader import build_loader
from loss import CleanContrastiveLoss
from model import Model, ArcMarginModel
from functions import tensor
from build import next_
import config


def train():
    net = Model().cuda()

    # metric_fc = ArcMarginModel()
    # metric_fc = nn.DataParallel(metric_fc)
    # metric_fc = metric_fc.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    criterion = CleanContrastiveLoss().cuda()

    train_loader, val_loader = build_loader()

    history = dict(
        loss=[],
        accuracy=[]
    )

    for epoch in range(config.EPOCHS):

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        for i in range(config.STEPS_PER_EPOCH):

            input1, label1, train_iter = next_(train_iter, train_loader)
            input2, label2, train_iter = next_(train_iter, train_loader)

            loss = criterion(net(input1), net(input2), label1, label2)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 200 == 0:
                input1, label1, train_iter = next_(val_iter, val_loader)
                input2, label2, train_iter = next_(val_iter, val_loader)

                prediction1 = torch.argmax(net(input1), dim=1)
                prediction2 = torch.argmax(net(input2), dim=1)

                accuracy = accuracy_score(
                    (net(input1) == net(input2).data),
                    (label1 == label2).data
                )

                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy)

                print(f'Epoch: {epoch} | train loss: {loss.item():.4f} | test accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    train()

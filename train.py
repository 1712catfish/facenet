import torch
from sklearn.metrics import accuracy_score

import config
from build import infinite_next
from functions import cpu
from loader import build_loader
from loss import CleanContrastiveLoss
from model import Model


def train():
    print('building net...')
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

        print('epoch ', epoch)

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        for i in range(config.STEPS_PER_EPOCH):

            (input1, label1), train_iter = infinite_next(train_iter, train_loader)
            (input2, label2), train_iter = infinite_next(train_iter, train_loader)

            loss = criterion(net(input1), net(input2), torch.eq(label1, label2))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                (input1, label1), train_iter = infinite_next(val_iter, val_loader)
                (input2, label2), train_iter = infinite_next(val_iter, val_loader)

                prediction1 = torch.argmax(net(input1), dim=1)
                prediction2 = torch.argmax(net(input2), dim=1)

                accuracy = accuracy_score(
                    cpu(torch.eq(prediction1, prediction2)),
                    cpu(torch.eq(label1, label2))
                )

                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy)

                print(f'step: ', i)
                print('loss: ', loss.item())
                print('accuracy: ', accuracy)
                print('prediction 1: ', cpu(prediction1[:5]))
                print('label 1: ', cpu(label1[:5]))
                print('prediction 2: ', cpu(prediction2[:5]))
                print('label 2: ', cpu(label2[:5]))
                print()


if __name__ == '__main__':
    train()

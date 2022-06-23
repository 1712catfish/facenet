import torch
from sklearn.metrics import accuracy_score

import config
from functions import cpu
from loader import build_loader, take
from loss import CleanContrastiveLoss
from model import Model
from plot import plot_history
from preprocessing import build_transform


def train():
    print('building net...')
    net = Model().to(config.DEVICE)

    # metric_fc = ArcMarginModel()
    # metric_fc = nn.DataParallel(metric_fc)
    # metric_fc = metric_fc.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)
    criterion = CleanContrastiveLoss().to(config.DEVICE)

    print('building loaders...')
    transform = build_transform()
    train_loader, val_loader = build_loader(transform)

    history = dict(
        loss=[],
        accuracy=[]
    )

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    print('training...')

    for epoch in range(config.EPOCHS):

        print('epoch', epoch)

        for i in range(config.STEPS_PER_EPOCH):
            input1, label1, input2, label2 = take(train_iter)

            output1 = net(input1)
            output2 = net(input2)

            loss = criterion(output1, output2, torch.eq(label1, label2))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if i % 10 == 0:
                input1, label1, input2, label2 = take(val_iter)

                prediction1 = torch.argmax(net(input1), dim=1)
                prediction2 = torch.argmax(net(input2), dim=1)

                accuracy = accuracy_score(
                    cpu(torch.eq(prediction1, prediction2)),
                    cpu(torch.eq(label1, label2))
                )

                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy)

                print('step:', '{:5d}'.format(i), end='  ')
                print('loss:', '%.6f' % (loss.item()), end='  ')
                print('accuracy:', '%.6f' % accuracy)
                # print('prediction 1:', prediction1[:5])
                # print('label 1:', label1[:5])
                # print('prediction 2:', prediction2[:5])
                # print('label 2:', label2[:5])

    plot_history(history)


if __name__ == '__main__':
    train()

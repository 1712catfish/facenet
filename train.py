# import config
# from functions import *
#
# def train(train_loader, val_loader, criterion, optimizer):
#
#     train_iter = iter(train_loader)
#     val_iter = iter(val_loader)
#
#     counter = []
#     loss_history = []
#     accuracy = []
#     iteration_number = 0
#
#     for epoch in range(config.EPOCHS):
#
#         x1, y2 = next(train_iter)
#         x1, y1 = tensor(x1), tensor(y1)
#         y_hat1 = model(x1)
#
#         x1, y2 = next(train_iter)
#         x1, y1 = tensor(x1), tensor(y1)
#         y_hat1 = model(x1)
#
#
#
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if i % config.LOG_EVERY == 0:
#                 lambda x: torch.argmax(x, dim=1)
#
#                 pred1, true1 = f(*next(val_iter))
#                 pred2, true2 = f(*next(val_iter))
#
#                 acc = sum(torch.logical_not(torch.logical_xor(pred1 == pred2, y1 == y2))) / float(y1.size(0))
#
#                 iteration_number += 10
#                 counter.append(iteration_number)
#                 loss_history.append(loss.item())
#                 accuracy.append(acc.item())
#                 print(f'Epoch: {epoch} | train loss: {loss.item():.4f} | test accuracy: {acc.item():.2f}')

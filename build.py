def infinite_next(train_iter, train_loader):
    try:
        return next(train_iter), train_iter
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter), train_iter

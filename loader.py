import random

import torch.utils.data as data
import torchvision
import torchvision.transforms as T

import config
from build import infinite_next
from functions import toss


class PairDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, pair_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.pair_rate = pair_rate
        self.cls = []

        prev_label = None
        for i, (_, label) in enumerate(self.imgs):
            if label != prev_label:
                self.cls.append(i)
                prev_label = label

    def __iter__(self):

        def iterator():

            if toss(self.pair_rate):
                k = random.randint(0, len(self.cls) - 2)
                i, j = random.sample(range(self.cls[k], self.cls[k + 1]), 2)
            else:
                k1, k2 = random.sample(range(len(self.cls) - 2), 2)
                i = random.randint(self.cls[k1], self.cls[k1 + 1])
                j = random.randint(self.cls[k2], self.cls[k2 + 1])

            yield super()[i], super()[j]

        return iterator()


class NaivePairDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, pair_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.co_pairs = []
        self.contra_pairs = []
        self.pair_rate = pair_rate

        for i, (_, label_i) in enumerate(self.imgs):
            print(i)
            for j, (_, label_j) in enumerate(self.imgs):
                if label_i == label_j:
                    self.co_pairs.append((i, j))
                else:
                    self.contra_pairs.append((i, j))

        print('finish generating pairs...')

    def __iter__(self):

        def iterator():
            co_iter = iter(self.co_pairs)
            contra_iter = iter(self.contra_pairs)

            if toss(self.pair_rate):
                (i, j), co_iter = infinite_next(co_iter, self.co_pairs)
            else:
                (i, j), contra_iter = infinite_next(contra_iter, self.contra_pairs)

            yield super()[i], super()[j]

        return iterator()


def build_loader():
    transform = T.Compose([
        T.CenterCrop(config.IMSIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    target_transform = T.Compose([
        # T.ToTensor(),
    ])

    print('building dataset...')
    train_ds = PairDataset(root=config.TRAIN_DIR,
                           transform=transform,
                           target_transform=target_transform)
    val_ds = PairDataset(root=config.VAL_DIR,
                         transform=transform,
                         target_transform=target_transform)

    print('building loader...')
    train_loader = data.DataLoader(train_ds,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=True, num_workers=2,
                                   drop_last=True)
    val_loader = data.DataLoader(val_ds,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=2,
                                 drop_last=True)

    return train_loader, val_loader

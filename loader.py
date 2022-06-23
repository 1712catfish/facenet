import torch.utils.data as data
import torchvision
import torchvision.transforms as T

import config
from build import infinite_next
from functions import toss


class PairDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, pair_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.co_pairs = []
        self.contra_pairs = []
        self.pair_rate = pair_rate

        print(type(self.imgs))

        for i1, (_, label1) in enumerate(self.imgs):
            for i2, (_, label2) in enumerate(self.imgs):
                if label1 == label2:
                    self.co_pairs.append((i1, i2))
                else:
                    self.contra_pairs.append((i1, i2))

    def __iter__(self):
        def iterator():
            co_iter = iter(self.co_pairs)
            contra_iter = iter(self.contra_pairs)

            if toss(self.pair_rate):
                co, co_iter = infinite_next(co_iter, self.co_pairs)
                yield co

            contra, contra_iter = infinite_next(contra_iter, self.contra_pairs)
            yield contra

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

    train_ds = PairDataset(root=config.TRAIN_DIR,
                           transform=transform,
                           target_transform=target_transform)
    val_ds = PairDataset(root=config.VAL_DIR,
                         transform=transform,
                         target_transform=target_transform)

    train_loader = data.DataLoader(train_ds,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=True, num_workers=2,
                                   drop_last=True)
    val_loader = data.DataLoader(val_ds,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True, num_workers=2,
                                 drop_last=True)

    return train_loader, val_loader

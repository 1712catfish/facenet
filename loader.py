import torchvision
import torch.utils.data as data
import torchvision.transforms as T
import config


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

    train_ds = torchvision.datasets.ImageFolder(root=config.TRAIN_DIR,
                                                transform=transform,
                                                target_transform=target_transform)
    val_ds = torchvision.datasets.ImageFolder(root=config.VAL_DIR,
                                              transform=transform,
                                              target_transform=target_transform)

    train_loader = data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    return train_loader, val_loader

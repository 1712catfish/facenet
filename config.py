import os

import torch

IMSIZE = 224
EMB_SIZE = 2048

EPOCHS = 1
STEPS_PER_EPOCH = 1000
LOG_EVERY = 200
BATCH_SIZE = 32
LR = 0.0003

TRAIN_DIR = '/content/train_lfw/merge'
VAL_DIR = '/content/train_lfw/merge'

NUM_CLASSES = len(os.listdir(TRAIN_DIR))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


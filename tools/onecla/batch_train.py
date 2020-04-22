from .train import main as train_main
from .utils import *
import os


def batch_train(cfg, gpus='1'):
    cfg = import_module(cfg)
    cfg.gpus = gpus
    train_main(cfg)


def main():
    batch_train('onecla/config/bottle/size_224x224_epoch_12.py')
    batch_train('onecla/config/bottle/size_1333x800_epoch_12.py')
    batch_train('onecla/config/bottle/size_224x224_epoch_52.py')
    batch_train('onecla/config/bottle/size_1333x800_epoch_52.py')


if __name__ == '__main__':
    main()

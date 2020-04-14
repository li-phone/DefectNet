from .train import main as train_main
from .utils import *
import os


def batch_train_with_size_224x224():
    loss_types = ['CrossEntropyLoss', ]
    for type in loss_types:
        cfg = import_module('config/coco_alcohol,size=224x224.py')
        cfg.gpus = '1'
        cfg.loss['type'] = type
        cfg.work_dir = os.path.join(cfg.work_dir,
                                    'coco_alcohol,loss={},seed=666,size=224x224'.format(type))
        cfg.resume_from = cfg.work_dir + '/latest.pth'
        train_main(cfg)


def batch_train_with_size_max_1333x800():
    loss_types = ['CrossEntropyLoss', ]
    for type in loss_types:
        cfg = import_module('onecla/config/coco_alcohol,size=max(1333x800).py')
        cfg.gpus = '1'
        cfg.loss['type'] = type
        cfg.work_dir = os.path.join(cfg.work_dir, 'coco_alcohol,loss={},seed=666,size=max(1333x800)'.format(type))
        cfg.resume_from = cfg.work_dir + '/latest.pth'
        train_main(cfg)


def batch_train_with_garbage_dataset():
    loss_types = ['CrossEntropyLoss', ]
    for type in loss_types:
        cfg = import_module('onecla/config/coco_garbage,size=224x224.py')
        cfg.gpus = '1'
        cfg.loss['type'] = type
        cfg.work_dir = os.path.join(cfg.work_dir, 'coco_garbage,loss={},seed=666,size=224x224'.format(type))
        cfg.resume_from = cfg.work_dir + '/latest.pth'
        train_main(cfg)


def main():
    batch_train_with_garbage_dataset()
    # batch_train_with_size_max_1333x800()
    # batch_train_with_size_224x224()


if __name__ == '__main__':
    main()

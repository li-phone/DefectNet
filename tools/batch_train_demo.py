import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mmcv.runner.runner import Runner

def train_models():
    from batch_train import BatchTrain

    # To make Figure 5.
    # train for finding best constant defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_constant_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=60, test_sleep_time=-60).find_best_constant_loss_weight()

    # To make Table 4.
    # train for exponent defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_exponent_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=60, test_sleep_time=-60).common_train()

    # train for inverse defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_inverse_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=60, test_sleep_time=-60).common_train()

    # train for linear defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_linear_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=60, test_sleep_time=-60).common_train()


def make_figures():
    pass


def make_tables():
    pass


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_models()
    make_figures()
    make_tables()


if __name__ == '__main__':
    main()

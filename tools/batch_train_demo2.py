import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mmcv.runner.runner import Runner


def train_models():
    from batch_train import BatchTrain

    # To make Table 2.
    # train for one-model method without background
    BatchTrain(cfg_path='../configs/bottle/one_model_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=-60).common_train()

    # train for one-model method with background
    BatchTrain(cfg_path='../configs/bottle/one_model_bg_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=-60).common_train()


def main():
    train_models()


if __name__ == '__main__':
    main()

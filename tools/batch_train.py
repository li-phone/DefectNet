# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
import copy
import json
import numpy as np
import os.path as osp
from batch_process import batch_train, batch_test, batch_infer

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


class BatchTrain(object):
    def __init__(self, cfg_path, data_mode='val', train_sleep_time=60, test_sleep_time=60 * 2):
        self.cfg_path = cfg_path
        self.cfg_dir = cfg_path[:cfg_path.rfind('/')]
        self.cfg_name = os.path.basename(cfg_path).split('.py')[0]
        self.data_mode = data_mode
        self.train_sleep_time = train_sleep_time
        self.test_sleep_time = test_sleep_time

    def find_best_constant_loss_weight(self):
        # To make Figure 5. The evaluation performance with increasing loss weight w.
        cfgs = []
        for weight in np.linspace(0, 2, 41):
            cfg = mmcv.Config.fromfile(self.cfg_path)
            cfg.model['dfn_balance']['init_weight'] = weight

            cfg.uid = 'const_weight={}'.format(weight)
            cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.uid)

            cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None
            cfgs.append(cfg)
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_test.txt')
        batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

        # batch_infer(cfgs)

    def common_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        cfg.first_model_cfg = None
        cfg.cfg_name = str(self.cfg_name)
        cfg.uid = str(self.cfg_name)
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(self.cfg_dir, str(self.cfg_name) + '_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)
        batch_infer(cfgs)


def main():
    pass


if __name__ == '__main__':
    main()

import mmcv
import glob
import os
import re
import numpy as np
from batch_train import BatchTrain


def models_train(test_status=10, data_type="fabric", const_weights=None):
    # train all models
    root = '../configs/{}/'.format(data_type)
    paths = glob.glob(os.path.join(root, 'baseline_model_*.py'))
    paths.sort()
    for cfg_path in paths:
        m = BatchTrain(cfg_path=cfg_path, data_mode='test', train_sleep_time=0, test_sleep_time=test_status)
        m.common_train()

    # test stacking tricks
    paths = glob.glob(os.path.join(root, '*_mst_*k*.py'))
    paths.sort()
    for cfg_path in paths:
        m = BatchTrain(cfg_path=cfg_path, data_mode='test', train_sleep_time=0, test_sleep_time=test_status)
        params = {'test_cfg': {'rcnn': mmcv.ConfigDict(
            score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.5), max_per_img=100)},
            'uid': "stacking tricks"
        }
        # m.common_train(**params)


def main():
    # train models
    models_train(data_type="tile")


if __name__ == '__main__':
    main()

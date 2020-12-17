import mmcv
import glob
import os
import re
import numpy as np
from batch_train import BatchTrain


def models_train(test_status=-10, data_type="fabric", const_weights=None):
    # train all models
    root = '../configs/{}/'.format(data_type)
    paths = glob.glob(os.path.join(root, 'defectnet/baseline_model.py'))
    paths.sort()
    for cfg_path in paths:
        m = BatchTrain(cfg_path=cfg_path, data_mode='test', train_sleep_time=0, test_sleep_time=test_status)
        m.common_train()

    # test stacking tricks
    paths = glob.glob(os.path.join(root, 'defectnet/*_mst_*_k*.py'))
    paths.sort()
    for cfg_path in paths:
        m = BatchTrain(cfg_path=cfg_path, data_mode='test', train_sleep_time=0, test_sleep_time=test_status)
        params = {'test_cfg': {'rcnn': mmcv.ConfigDict(
            score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.5), max_per_img=100)},
            'uid': "stacking tricks"
        }
        m.common_train(**params)

    # test different softnms thresholds
    iou_thrs = np.linspace(0.1, 0.9, 9)
    for iou_thr in iou_thrs:
        softnms_model = BatchTrain(cfg_path='../configs/{}/baseline_model.py'.format(data_type),
                                   data_mode='test', train_sleep_time=0, test_sleep_time=test_status)
        params = {'test_cfg': {'rcnn': mmcv.ConfigDict(
            score_thr=0.05, nms=dict(type='soft_nms', iou_thr=iou_thr), max_per_img=100)},
            'uid': iou_thr
        }
        softnms_model.common_train(**params)


def main():
    # train models
    models_train(data_type="bag_tricks")
    # test models
    models_train(test_status=60 * 1, data_type="bag_tricks")
    pass


if __name__ == '__main__':
    main()

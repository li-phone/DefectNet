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
from train import main as train_main
from defect_test import main as test_main
from infer import main as infer_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


def hint(wav_file='./wav/qq.wav', n=5):
    import pygame
    for i in range(n):
        pygame.mixer.init()
        pygame.mixer.music.load(wav_file)
        pygame.mixer.music.set_volume(0.5)
        pygame.mixer.music.play()


def batch_infer(cfgs):
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))
        infer_params = dict(
            config=cfg,
            resume_from=osp.join(cfg.work_dir, 'epoch_12.pth'),
            infer_object=cfg.data['test']['ann_file'],
            img_dir=cfg.data['test']['img_prefix'],
            work_dir=cfg.work_dir,
            submit_out=osp.join(cfg.work_dir, '{}_submit,epoch={},.json'.format(cfg_name, 12)),
            have_bg=False,
        )
        infer_main(**infer_params)
        print('{} infer successfully!'.format(cfg_name))
        hint()


def batch_train(cfgs, sleep_time=0, detector=True):
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        # train
        train_params = dict(config=cfg, detector=detector)
        train_main(**train_params)
        print('{} train successfully!'.format(cfg_name))
        hint()
        time.sleep(sleep_time)


def baseline_one_model_train_with_background():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['baseline.py', ]

    # watch train effects using different base cfg
    ratios = [1]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
        cfg.cfg_name = 'baseline_one_model'
        cfg.uid = None
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'baseline_one_model,background=Yes')
        cfg.data['train']['ignore_ids'] = None
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfgs.append(cfg)
    batch_train(cfgs, sleep_time=60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'baseline_one_model_test,background=Yes,.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def baseline_one_model_train_no_background():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['baseline.py', ]

    # watch train effects using different base cfg
    ratios = [1]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = 'baseline_one_model'
        cfg.uid = 'background=No'
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'baseline_one_model,background=No')

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfgs.append(cfg)
    batch_train(cfgs, sleep_time=60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'baseline_one_model_test,background=No,.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def batch_fixed_defect_finding_weight_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    # watch train effects using different base cfg
    ratios = np.linspace(0.1, 0.2, 6)
    # ratios = np.linspace(0., 0.1, 6)
    # ratios = np.append(ratios, np.linspace(0.3, 1.9, 9))
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)
        cfg.model['dfn_weight'] = n

        cfg.cfg_name = 'different_fixed_dfn_weight'
        cfg.uid = n
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name,
            'different_fixed_dfn_weight,weight={:.2f},loss={}'.format(
                n, cfg.model['backbone']['loss_cls']['type']))

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfgs.append(cfg)
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(
        cfg_dir,
        'different_dfn_weight_test,loss={},weight=0.00-2.00,.txt'.format(cfgs[0].model['backbone']['loss_cls']['type']))
    # batch_test(cfgs, save_path, 60 * 2, mode='val')


def two_model_first_model_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['first_model.py', ]

    # watch train effects using different base cfg
    ratios = [1]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = 'two_model'
        cfg.uid = 'model=first,loss=CrossEntropyLoss'
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'model=first')

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfg.total_epochs = 12
        cfgs.append(cfg)
    batch_train(cfgs, sleep_time=60 * 2, detector=False)
    from batch_test import cls_batch_test
    save_path = os.path.join(cfg_dir, 'two_model_first_model_test,model=first,.txt')
    cls_batch_test(cfgs, save_path, 60 * 2, mode='val')


def garbage_baseline_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['garbage.py', ]

    # watch train effects using different base cfg
    ratios = [1]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.data['imgs_per_gpu'] = 2
        cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

        cfg.cfg_name = 'garbage_baseline'
        cfg.uid = 'garbage_baseline'
        cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, 'garbage_baseline')

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None
        cfgs.append(cfg)
    # batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage_test.txt')
    # batch_test(cfgs, save_path, 60 * 2, mode='test')
    batch_infer(cfgs)


def main():
    # garbage_baseline_train()

    # one model
    # baseline_one_model_train_with_background()
    # baseline_one_model_train_no_background()

    # two model
    # two_model_first_model_train()

    # defect network
    batch_fixed_defect_finding_weight_train()


if __name__ == '__main__':
    main()

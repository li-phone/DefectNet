# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
from mmcv import ConfigDict
import copy
import json
import numpy as np
import os.path as osp
from train import main as train_main
from defect_test import main as test_main
from infer import main as infer_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)
DATA_NAME = 'garbage'
DATA_MODE = 'val'


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
            submit_out=osp.join(cfg.work_dir, '{}_submit,epoch_{}.json'.format(cfg_name, 12)),
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


def baseline_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'mode=baseline'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)
    # from batch_train import batch_infer
    # batch_infer(cfgs)


def anchor_ratios_cluster_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    # new added
    from tricks.data_cluster import anchor_cluster
    anchor_ratios = anchor_cluster(cfg.data['train']['ann_file'], n=6)
    cfg.model['rpn_head']['anchor_ratios'] = list(anchor_ratios)

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'anchor_cluster=6'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)
    # from batch_train import batch_infer
    # batch_infer(cfgs)


def larger_lr_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] *= 1.

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'lr={:.2f}'.format(cfg.optimizer['lr'])
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def twice_epochs_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.total_epochs *= 2

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'epoch=2x'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def OHEMSampler_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    for rcnn in cfg.train_cfg['rcnn']:
        rcnn['sampler'] = dict(
            type='OHEMSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True)

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'sampler=OHEMSampler'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def multi_scale_train(img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True):
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.train_pipeline[2] = mmcv.ConfigDict(
        type='Resize', img_scale=img_scale, ratio_range=ratio_range,
        multiscale_mode=multiscale_mode, keep_ratio=keep_ratio)
    sx = int(np.mean([v[0] for v in img_scale]))
    sy = int(np.mean([v[1] for v in img_scale]))
    cfg.test_pipeline[1]['img_scale'] = [(sx, sy)]

    cfg.data['train']['pipeline'] = cfg.train_pipeline
    cfg.data['val']['pipeline'] = cfg.test_pipeline
    cfg.data['test']['pipeline'] = cfg.test_pipeline

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'img_scale={},multiscale_mode={},ratio_range={}' \
        .format(str(img_scale), str(multiscale_mode), str(ratio_range))
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def load_pretrain_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.load_from = ''

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'load_from={}'.format(os.path.basename(cfg.load_from))
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def backbone_dcn_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.model['backbone']['dcn'] = dict(  # 在最后三个block加入可变形卷积
        modulated=False, deformable_groups=1, fallback_on_stride=False)
    cfg.model['backbone']['stage_with_dcn'] = (False, True, True, True)

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'backbone_dcn=True'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def iou_thr_train(iou_thrs=[0.5, 0.6, 0.7]):
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    for i, rcnn in enumerate(cfg.train_cfg['rcnn']):
        rcnn['assigner']['pos_iou_thr'] = iou_thrs[i]
        rcnn['assigner']['neg_iou_thr'] = iou_thrs[i]
        rcnn['assigner']['min_pos_iou'] = iou_thrs[i]

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'rcnn_iou_thrs={}'.format(str(iou_thrs))
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def SWA_train():
    import torch
    import os
    import mmcv
    from mmdet.models import build_detector

    def get_model(config, model_dir):
        model = build_detector(config.model, test_cfg=config.test_cfg)
        checkpoint = torch.load(model_dir)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=True)
        return model

    def model_average(modelA, modelB, alpha):
        # modelB占比 alpha
        for A_param, B_param in zip(modelA.parameters(), modelB.parameters()):
            A_param.data = A_param.data * (1 - alpha) + alpha * B_param.data
        return modelA

    def swa(cfg, epoch_inds, alpha=0.7):
        ###########################注意，此py文件没有更新batchnorm层，所以只有在mmdetection默认冻住BN情况下使用，如果训练时BN层被解冻，不应该使用此py　＃＃＃＃＃
        #########逻辑上会　score　会高一点不会太多，需要指定的参数是　[config_dir , epoch_indices ,  alpha]　　######################
        if isinstance(cfg, str):
            config = mmcv.Config.fromfile(cfg)
        else:
            config = cfg
        work_dir = config.work_dir
        model_dir_list = [os.path.join(work_dir, 'epoch_{}.pth'.format(epoch)) for epoch in epoch_inds]

        model_ensemble = None
        for model_dir in model_dir_list:
            if model_ensemble is None:
                model_ensemble = get_model(config, model_dir)
            else:
                model_fusion = get_model(config, model_dir)
                model_ensemble = model_average(model_ensemble, model_fusion, alpha)

        checkpoint = torch.load(model_dir_list[-1])
        checkpoint['state_dict'] = model_ensemble.state_dict()
        ensemble_path = os.path.join(work_dir, 'epoch_ensemble.pth')
        torch.save(checkpoint, ensemble_path)
        return ensemble_path

    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'SWA=[10,11,12]'

    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + 'mode=baseline')

    ensemble_path = swa(cfg, [10, 11, 12])
    cfg.resume_from = ensemble_path
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    # batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def anchor_scales_cluster_train():
    from tricks.data_cluster import box_cluster
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    # new added
    boxes = box_cluster(cfg.data['train']['ann_file'], n=5)
    anchor_scales = np.sqrt(boxes[0][0] * boxes[0][1])
    anchor_scales = min(anchor_scales * 1333 / 2446, anchor_scales * 800 / 1000) / 4
    anchor_scales = int(anchor_scales)
    cfg.model['rpn_head']['anchor_scales'] = list([anchor_scales])

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    if anchor_scales == 8:
        cfg.uid = 'mode=baseline'
        print('This trick is the same with baseline model.')
    else:
        cfg.uid = 'anchor_scales_cluster=6'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def global_context_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.model['bbox_roi_extractor']['global_context'] = True

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'global_context=True'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def score_thr_train(score_thr=0.02):
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
    cfg.test_cfg['rcnn']['score_thr'] = score_thr

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'

    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + 'mode=baseline')
    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfg.uid = 'score_thr={}'.format(score_thr)
    cfgs = [cfg]
    # batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)


def other_cfg_train():
    cfg_dir = '../other_cfgs/'
    cfgs = [mmcv.Config.fromfile('../other_cfgs/cascade.py')]
    cfgs[0].first_model_cfg = None
    cfgs[0].uid = 'cutout'
    cfgs[0].resume_from = cfgs[0].work_dir + '/latest.pth'
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, 'garbage' + '_test.txt')
    # batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)
    from batch_train import batch_infer
    batch_infer(cfgs)


def joint_train():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = [DATA_NAME + '.py', ]

    cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

    # 0.810 (img_scale = [1920, 1080]) ==> (img_scale = [(1920, 1080), (1333, 800)], multiscale_mode='range')
    img_scale = [(1920, 1080), (1333, 800)]
    cfg.train_pipeline[2] = mmcv.ConfigDict(
        type='Resize', img_scale=img_scale, ratio_range=None,
        multiscale_mode='range', keep_ratio=True)
    sx = int(max([v[0] for v in img_scale]))
    sy = int(max([v[1] for v in img_scale]))
    cfg.test_pipeline[1]['img_scale'] = [(sx, sy)]

    cfg.data['train'] = cfg.train_pipeline
    cfg.data['val'] = cfg.test_pipeline
    cfg.data['test'] = cfg.test_pipeline

    # 0.819
    cfg.model['backbone']['dcn'] = dict(  # 在最后三个block加入可变形卷积
        modulated=False, deformable_groups=1, fallback_on_stride=False)
    cfg.model['backbone']['stage_with_dcn'] = (False, True, True, True)

    # 0.822
    # global context
    cfg.model['bbox_roi_extractor']['global_context'] = True

    # # 0.746
    # from tricks.data_cluster import anchor_cluster
    # anchor_ratios = anchor_cluster(cfg.data['train']['ann_file'], n=6)
    # cfg.model['rpn_head']['anchor_ratios'] = list(anchor_ratios)

    # 0.???
    # focal loss for rcnn
    # for head in cfg.model['bbox_head']:
    #     head['loss_cls'] = dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0)

    cfg.data['imgs_per_gpu'] = 2
    cfg.optimizer['lr'] = cfg.optimizer['lr'] / 8 * (cfg.data['imgs_per_gpu'] / 2)

    cfg.cfg_name = DATA_NAME + '_baseline'
    cfg.uid = 'mode=joint_train+multiscale_mode=range'
    cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name, cfg.cfg_name + ',' + cfg.uid)

    cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
    if not os.path.exists(cfg.resume_from):
        cfg.resume_from = None

    cfgs = [cfg]
    batch_train(cfgs, sleep_time=0 * 60 * 2)
    from batch_test import batch_test
    save_path = os.path.join(cfg_dir, DATA_NAME + '_test.txt')
    batch_test(cfgs, save_path, 60 * 2, mode=DATA_MODE)
    from batch_train import batch_infer
    batch_infer(cfgs)


def main():
    joint_train()
    # other_cfg_train()

    # trick 0: baseline
    # baseline_train()

    # trick 1: anchor cluster
    # anchor_ratios_cluster_train()

    # trick 2: larger lr
    # larger_lr_train()

    # trick 3: 2x epochs
    # twice_epochs_train()

    # trick 12: global context
    # global_context_train()

    # trick 5:
    # OHEMSampler_train()

    # trick 6: multiple scales train
    # multi_scale_train()

    # trick 7: load pretrained model train
    # load_pretrain_train()

    # trick 8: backbone_dcn_train
    # backbone_dcn_train()

    # trick 9:
    # iou_thr_train([0.5, 0.6, 0.7])
    # iou_thr_train([0.6, 0.7, 0.8])
    # iou_thr_train([0.4, 0.5, 0.6])

    # trick 10: swa ensemble
    # SWA_train()

    # trick 11:
    # anchor_scales_cluster_train()

    # trick 13: low score_thr
    # score_thr_train()

    # from analyze_data import phrase_json
    # phrase_json('../config_alcohol/cascade_rcnn_r50_fpn_1x/' + DATA_NAME + '_test.json')
    # pass


if __name__ == '__main__':
    main()

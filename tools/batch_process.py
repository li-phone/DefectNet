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

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


def batch_train(cfgs, sleep_time=0, detector=True):
    for cfg in tqdm(cfgs):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        # train
        train_params = dict(config=cfg, detector=detector)
        train_main(**train_params)
        print('{} train done!'.format(cfg_name))
        time.sleep(sleep_time)


def eval_report(rpt_txt, rpts, cfg, uid=None, mode='val'):
    save_dir = rpt_txt[:rpt_txt.rfind('/')]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(rpt_txt, 'a+') as fp:
        head = '\n\n{} {}, uid: {}, mode: {} {}\n'.format('=' * 36, cfg, uid, mode, '=' * 36)
        fp.write(head)
        for k1, v1 in rpts.items():
            fp.write(k1 + ':\n')  # bbox

            fp.write('log' + ':\n')  # log
            for k2, v2 in v1['log'].items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        fp.write('\n' + k3 + ':\n' + str(v3) + '\n')
                else:
                    fp.write('\n' + k2 + ':\n' + str(v2) + '\n')
    json_txt = rpt_txt[:-4]
    with open(json_txt + '.json', 'a+') as fp:
        if uid is None:
            uid = cfg
        jstr = json.dumps(dict(cfg=cfg, uid=uid, mode=mode, data=rpts))
        fp.write(jstr + '\n')


def batch_test(cfgs, save_dir, sleep_time=0, mode='test', json_out_heads=None):
    save_name = os.path.basename(save_dir)
    save_name = save_name[:save_name.rfind('.')]
    save_dir = save_dir.replace('\\', '/')
    save_dir = save_dir[:save_dir.rfind('/')]
    for i, cfg in tqdm(enumerate(cfgs)):
        cfg_name = os.path.basename(cfg.work_dir)
        print('\ncfg: {}'.format(cfg_name))

        json_out_head = ''
        if json_out_heads is not None:
            if isinstance(json_out_heads, list):
                json_out_head = json_out_heads[i]
            elif isinstance(json_out_heads, str):
                json_out_head = json_out_heads
        eval_test_params = dict(
            config=cfg,
            checkpoint=osp.join(cfg.work_dir, 'latest.pth'),
            json_out=osp.join(cfg.work_dir, 'mode={}+{}.json'.format(mode, json_out_head)),
            mode=mode,
        )
        report = test_main(**eval_test_params)
        eval_report(osp.join(save_dir, save_name + '.txt'), report, cfg=cfg_name, uid=cfg.uid, mode=mode)
        print('{} eval test done!'.format(cfg_name))
        time.sleep(sleep_time)

# batch train
from tqdm import tqdm
import os
import time
from mmdet.core import coco_eval
import mmcv
import copy
import json
import numpy as np
from pycocotools.coco import COCO
import os.path as osp
import pandas as pd
from defect_test import main as test_main

BASH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASH_DIR)


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
            json_out=osp.join(cfg.work_dir, 'mode={},{}.json'.format(mode, json_out_head)),
            mode=mode,
        )
        report = test_main(**eval_test_params)
        eval_report(osp.join(save_dir, save_name + '.txt'), report, cfg=cfg_name, uid=cfg.uid, mode=mode)
        print('{} eval test successfully!'.format(cfg_name))
        time.sleep(sleep_time)



def different_defect_finding_weight_test():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    ratios = np.linspace(0., 0.1, 6)
    ratios = np.append(ratios, np.linspace(0.2, 2, 10))
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.model['dfn_weight'] = n

        cfg.cfg_name = 'different_fixed_dfn_weight'
        cfg.uid = n
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name,
            'different_fixed_dfn_weight,weight={:.2f},loss={}'.format(
                n, cfg.model['backbone']['loss_cls']['type']))

        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        cfgs.append(cfg)

    save_path = os.path.join(
        cfg_dir,
        'different_dfn_weight_test,loss={},weight=0.00-2.00,.txt'.format(cfgs[0].model['backbone']['loss_cls']['type']))
    batch_test(cfgs, save_path, 60 * 2, mode='test')
    batch_test(cfgs, save_path, 60 * 2, mode='val')


def different_normal_image_ratio_test():
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['defectnet.py', ]

    def del_images(dataset, img_ids):
        for img_id in img_ids:
            for i, image in enumerate(dataset['images']):
                if image['id'] == img_id:
                    dataset['images'].pop(i)
            for i, ann in enumerate(dataset['annotations']):
                if ann['image_id'] == img_id:
                    dataset['annotations'].pop(i)
        return dataset

    def cls_dataset(coco):
        if isinstance(coco, str):
            coco = COCO(coco)
        normal_ids, defect_ids = [], []
        for image in coco.dataset['images']:
            img_id = image['id']
            ann_ids = coco.getAnnIds(img_id)
            anns = coco.loadAnns(ann_ids)
            cnt = 0
            for i, ann in enumerate(anns):
                if ann['category_id'] != 0:
                    cnt += 1
            if cnt == 0:
                normal_ids.append(img_id)
            else:
                defect_ids.append(img_id)
        return normal_ids, defect_ids

    coco = COCO('/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.json')
    normal_ids, defect_ids = cls_dataset(coco)
    defect_ids = pd.DataFrame({'id': defect_ids})
    dataset = coco.dataset

    ann_files, uids = [], []
    choose_num = int(defect_ids.shape[0] * 0.05)
    save_json_dir = '/home/liphone/undone-work/data/detection/alcohol/annotations/normal_ratios'
    if not os.path.exists(save_json_dir):
        os.makedirs(save_json_dir)
    while defect_ids.shape[0] > 0:
        ratio = len(normal_ids) / defect_ids.shape[0]
        uids.append(ratio)
        fn = os.path.join(save_json_dir, 'test_set_normal_ratio={}.json'.format(ratio))
        with open(fn, 'w')as fp:
            json.dump(dataset, fp)
        ann_files.append(fn)

        img_ids = defect_ids.sample(n=min(choose_num, defect_ids.shape[0]), random_state=666)
        dataset = del_images(dataset, np.array(img_ids['id']))
        defect_ids = defect_ids.drop(img_ids.index)

    cfgs = []
    for i, ann_file in enumerate(ann_files):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))
        cfg.model['dfn_weight'] = 0.1
        cfg.data['test']['ann_file'] = ann_file

        cfg.cfg_name = 'fixed_defect_finding_weight'
        cfg.uid = 0.1
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'fixed_defect_finding_weight={:.1f}'.format(cfg.uid))
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        cfgs.append(cfg)

    batch_test(cfgs, cfg_dir + '/different_normal_image_ratio_test.txt', 60, mode='test')


def two_model_test():
    import torchvision.models.resnet
    cfg_dir = '../config_alcohol/cascade_rcnn_r50_fpn_1x'
    cfg_names = ['baseline.py', ]

    # watch train effects using different base cfg
    first_model_cfg = [
        'onecla/config/coco_alcohol,size=max(1333x800).py',
        'onecla/config/coco_alcohol,size=max(1333x800).py',
        'onecla/config/coco_alcohol,size=224x224.py',
        'onecla/config/coco_alcohol,size=224x224.py',
    ]
    first_code_py = 'onecla/infer.py'
    first_model_path = [
        'onecla/work_dirs/coco_alcohol/resnet50/coco_alcohol,loss=CrossEntropyLoss,seed=666,size=max(1333x800)/epoch_000011.pth',
        'onecla/work_dirs/coco_alcohol/resnet50/coco_alcohol,loss=CrossEntropyLoss,seed=666,size=max(1333x800)/epoch_000051.pth',
        'onecla/work_dirs/coco_alcohol/resnet50/coco_alcohol,loss=CrossEntropyLoss,seed=666,size=224x224/epoch_000011.pth',
        'onecla/work_dirs/coco_alcohol/resnet50/coco_alcohol,loss=CrossEntropyLoss,seed=666,size=224x224/epoch_000051.pth',
    ]
    sizes = ['1333x800', '1333x800', '224x224', '224x224', ]
    epochs = [12, 52, 12, 52]

    ratios = [1, 2, 3, 4]
    ns = ratios
    cfgs = []
    for i, n in enumerate(ns):
        cfg = mmcv.Config.fromfile(os.path.join(cfg_dir, cfg_names[0]))

        cfg.first_model_cfg = first_model_cfg[i]
        cfg.first_code_py = first_code_py
        cfg.first_model_path = first_model_path[i]

        cfg.cfg_name = 'baseline_one_model'
        cfg.uid = 'size={},epoch={},background=No'.format(sizes[i], epochs[i])
        cfg.work_dir = os.path.join(
            cfg.work_dir, cfg.cfg_name, 'baseline_one_model,background=No')
        cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
        cfgs.append(cfg)
    save_path = os.path.join(cfg_dir, 'two_model_test,background=No,.txt')
    batch_test(cfgs, save_path, 60 * 2, mode='test')


def main():
    # one model
    # one_model_no_background_test()
    # one_model_with_background_test()
    # different_threshold_no_background_test()

    # two model
    # two_model_test()

    # defect network
    different_defect_finding_weight_test()
    # different_normal_image_ratio_test()

    pass


if __name__ == '__main__':
    main()

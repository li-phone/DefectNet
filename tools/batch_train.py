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
from pycocotools.coco import COCO
from batch_process import batch_train, batch_test

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

    def find_best_constant_loss_weight(self, const_weights=None):
        # To make Figure 5. The evaluation performance with increasing loss weight w.
        cfgs = []
        if const_weights is None:
            const_weights = np.linspace(0, 2, 41)
        for weight in const_weights:
            cfg = mmcv.Config.fromfile(self.cfg_path)
            cfg.model['dfn_balance']['init_weight'] = weight

            cfg.uid = weight
            cfg.cfg_name = 'const_weight={:.2f}'.format(cfg.uid)
            cfg.work_dir = os.path.join(cfg.work_dir, cfg.cfg_name)

            cfg.resume_from = os.path.join(cfg.work_dir, 'epoch_12.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None
            cfgs.append(cfg)
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(cfgs[0].work_dir, str(self.cfg_name) + '_find_best_weight_test.txt')
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

    def common_train(self):
        cfg = mmcv.Config.fromfile(self.cfg_path)
        cfg.first_model_cfg = None
        cfg.cfg_name = str(self.cfg_name)
        cfg.uid = str(self.cfg_name)
        cfg.resume_from = os.path.join(cfg.work_dir, 'epoch_12.pth')
        if not os.path.exists(cfg.resume_from):
            cfg.resume_from = None

        cfgs = [cfg]
        batch_train(cfgs, sleep_time=self.train_sleep_time)
        save_path = os.path.join(cfg.work_dir, str(self.cfg_name) + '_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode)

    def score_threshold_test(self):
        cfgs = []
        json_out_heads = []
        for score in np.linspace(0, 0.99, 100):
            cfg = mmcv.Config.fromfile(self.cfg_path)
            cfg.test_cfg['rcnn']['score_thr'] = score

            json_out_head = 'score_thr={:.2f}'.format(score)
            json_out_heads.append(json_out_head)

            cfg.uid = 'score_thr={}'.format(score)

            cfg.resume_from = os.path.join(cfg.work_dir, 'epoch_12.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None
            cfgs.append(cfg)
        save_path = os.path.join(cfgs[0].work_dir,
                                 str(self.cfg_name) + '_score_threshold_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode, json_out_heads=json_out_heads)

    def two_model_test(self, data_type="fabric"):

        # watch train effects using different base cfg
        first_model_cfg = [
            'onecla/config/{}/size_224x224_epoch_12.py'.format(data_type),
            'onecla/config/{}/size_224x224_epoch_52.py'.format(data_type),
            'onecla/config/{}/size_1333x800_epoch_12.py'.format(data_type),
            'onecla/config/{}/size_1333x800_epoch_52.py'.format(data_type),
        ]
        first_model_path = [
            '../work_dirs/{}/resnet50/size_224x224_epoch_12/epoch_000011.pth'.format(data_type),
            '../work_dirs/{}/resnet50/size_224x224_epoch_52/epoch_000051.pth'.format(data_type),
            '../work_dirs/{}/resnet50/size_1333x800_epoch_12/epoch_000011.pth'.format(data_type),
            '../work_dirs/{}/resnet50/size_1333x800_epoch_52/epoch_000051.pth'.format(data_type),
        ]
        first_code_py = 'onecla/infer.py'

        cfgs, json_out_heads = [], []
        for i, v in enumerate(first_model_cfg):
            cfg = mmcv.Config.fromfile(self.cfg_path)

            cfg.first_model_cfg = first_model_cfg[i]
            cfg.first_model_path = first_model_path[i]
            cfg.first_code_py = first_code_py

            basename = os.path.basename(v)
            basename = basename[:basename.rfind('.py')]
            json_out_head = basename
            json_out_heads.append(json_out_head)

            cfg.uid = os.path.basename(basename)

            cfg.resume_from = os.path.join(cfg.work_dir, 'latest.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None
            cfgs.append(cfg)
        save_path = os.path.join(cfgs[0].work_dir, str(self.cfg_name) + '_two_model_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode, json_out_heads=json_out_heads)

    def normal_proportion_test(self):

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

        def make_proportion_json(ann_file, proportion, save_name, random_state=666):
            from extends.cocoutils.coco_split import get_coco_by_imgids
            import pandas as pd
            coco = COCO(ann_file)
            normal_ids, defect_ids = cls_dataset(coco)
            prop = len(normal_ids) / (len(normal_ids) + len(defect_ids))
            normal_id_df = pd.DataFrame({'id': normal_ids})
            defect_id_df = pd.DataFrame({'id': defect_ids})
            if proportion <= prop:
                # increase normal images
                a = proportion * len(defect_ids) / (1 - proportion)
                normal_id_df = normal_id_df.sample(n=int(a), random_state=random_state)
                defect_ids.extend(list(normal_id_df['id']))
                dataset = get_coco_by_imgids(coco, defect_ids)
            else:
                # decrease defective images
                b = len(normal_ids) / proportion - len(normal_ids)
                defect_id_df = defect_id_df.sample(n=int(b), random_state=random_state)
                normal_ids.extend(list(defect_id_df['id']))
                dataset = get_coco_by_imgids(coco, normal_ids)
            with open(save_name, 'w')as fp:
                json.dump(dataset, fp)
            return len(normal_id_df), len(defect_id_df)

        cfgs, json_out_heads = [], []
        for proportion in np.linspace(0., 1., 101):
            cfg = mmcv.Config.fromfile(self.cfg_path)
            ann_file = os.path.join(cfg.work_dir, 'proportion/normal_proportion={:.2f}_test.json'.format(proportion))
            prop_cnt = [None, None]
            if True or not os.path.exists(ann_file):
                ann_file = ann_file.replace('\\', '/')
                save_dir = ann_file[:ann_file.rfind('/')]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # try to keep as many images as possible in keeping proportion
                prop_cnt = make_proportion_json(cfg.data['test']['ann_file'], proportion, ann_file)
            cfg.data['test']['ann_file'] = ann_file

            json_out_head = 'proportion={:.2f}'.format(proportion)
            json_out_heads.append(json_out_head)

            cfg.uid = 'proportion={}, normal_cnt={}, defect_cnt={}'.format(proportion, prop_cnt[0], prop_cnt[1])

            cfg.resume_from = os.path.join(cfg.work_dir, 'epoch_12.pth')
            if not os.path.exists(cfg.resume_from):
                cfg.resume_from = None
            cfgs.append(cfg)
        save_path = os.path.join(cfgs[0].work_dir,
                                 str(self.cfg_name) + '_normal_proportion_{}.txt'.format(self.data_mode))
        if self.test_sleep_time >= 0:
            batch_test(cfgs, save_path, self.test_sleep_time, mode=self.data_mode, json_out_heads=json_out_heads)


def main():
    pass


if __name__ == '__main__':
    main()

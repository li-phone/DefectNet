import torch
from torchvision.models import resnet34, resnet101
import argparse
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
from torchnet import meter
from sklearn.metrics.classification import classification_report
import pandas as pd
from tqdm import tqdm
import time
from torch.utils.data import *
from .build_network import *
from .utils import *
from .data_reader import DataReader, collate_fn, transform_compose


class Inference(object):
    def __init__(self, cfg, model_path=None):
        if isinstance(cfg, str):
            cfg = import_module(cfg)
        self.cfg = cfg
        if model_path is not None:
            self.cfg.resume_from = model_path
        self.cfg.gpus = prase_gpus(self.cfg.gpus)
        self.model = build_network(**self.cfg.model_config, gpus=self.cfg.gpus)
        self.model, optimizer, lr_scheduler, last_epoch = resume_network(self.model, self.cfg)
        self.transform_compose = transform_compose
        self.mode = 'test'
        self.img_scale = self.cfg.dataset[self.mode]['img_scale']
        self.keep_ratio = self.cfg.dataset[self.mode]['keep_ratio']

    def infer(self, img_paths):
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        results, data_times, infer_times = [], [], []
        for img_path in img_paths:
            torch.cuda.synchronize()
            start_time = time.time()
            img = self.transform_compose(img_path, img_scale=self.img_scale, keep_ratio=self.keep_ratio, mode=self.mode)
            img = img.unsqueeze(dim=0).cuda()
            torch.cuda.synchronize()
            end_time = time.time()
            data_times.append(end_time - start_time)

            torch.cuda.synchronize()
            start_time = time.time()
            out = self.model(img)
            score = out.softmax(dim=1)
            label = score.argmax(dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            infer_times.append(end_time - start_time)

            results.append(int(label))

        return results, data_times, infer_times


def infer(model, cfg, cuda=True):
    img_dirs = [r['img_prefix'] for r in cfg.dataset['test']]
    data_reader = DataReader(None, img_dirs, transform=None)
    data_loader = DataLoader(data_reader, collate_fn=collate_fn, **cfg.val_data_loader)
    y_pred = []
    model.eval()
    for step, (data) in tqdm(enumerate(data_loader)):
        inputs = torch.stack(data)
        if cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        outs = nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(outs, dim=1)
        # pred = outs[:, 0]
        y_pred.extend(list(pred.cpu().detach().numpy()))
        # break
    model.train()
    ids = [os.path.basename(x) for x in data_reader.image_paths]
    ids = [x.split('.')[0] for x in ids]
    ids = [int(x) for i, x in enumerate(ids)]
    return pd.DataFrame(data=dict(ids=ids[:len(y_pred)], label=y_pred))


def main(cfg):
    mkdirs(cfg.work_dir)

    cfg.gpus = prase_gpus('1')
    model = build_network(**cfg.model_config, gpus=cfg.gpus)
    model, optimizer, lr_scheduler, last_epoch = resume_network(model, cfg)

    submit_df = infer(model, cfg)
    save_name = os.path.join(cfg.work_dir, '{}_epoch_{}_submit.csv'.format(cfg.model_config['name'], last_epoch))
    submit_df.to_csv(save_name, index=False, header=False)
    logger.info('infer successfully!')


if __name__ == '__main__':
    pcfg = import_module("cfg.py")
    dcfg = import_module(pcfg.dataset_cfg_path)
    main(dcfg)

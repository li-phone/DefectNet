import importlib
import os
import json
import numpy as np
import math
import sys
import pandas as pd
import datetime
import logging
import time
import torch


# ===================================== Common Functions =====================================
def check_input(imgs, targets):
    import cv2 as cv
    import numpy as np
    for idx, target in enumerate(targets):
        img = imgs[idx]
        img = img.permute(1, 2, 0)  # C x H x W --> H x W x C
        image = np.array(img.cpu()).copy()
        image = np.array(image[..., ::-1])  # RGB --> BGR
        org = tuple([int(_ / 2) for _ in image.shape])
        cv.putText(image, str(target), tuple(org[:2][::-1]), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv.imshow("img", image)
        cv.waitKey(0)


def get_et_time(t):
    t = int(t)
    ets = []
    # s
    ets.append(t % 60)
    # m
    t = t // 60
    ets.append(t % 60)
    # h
    t = t // 60
    ets.append(t % 24)
    # d
    t = t // 24
    if t != 0:
        ets.append(t)
    ets.reverse()
    ets = ['{:02d}'.format(_) for _ in ets]
    return ':'.join(ets)


def save_model(d, epoch, save_dir):
    save_name = os.path.join(save_dir, 'epoch_{:06d}.pth'.format(epoch))
    torch.save(d, save_name)

    dst = os.path.join(save_dir, 'latest.pth')
    if os.path.islink(dst):
        os.remove(dst)
    os.symlink(save_name, dst)


def import_module(path):
    py_idx = path.rfind('.py')
    if py_idx != -1:
        path = path[:py_idx]
    _module_path = path.replace('\\', '/')
    _module_path = _module_path.replace('/', '.')
    return importlib.import_module(_module_path)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_date_str():
    time_str = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    return time_str


def prase_gpus(gpus):
    try:
        _gpus = gpus.split('-')
        if len(_gpus) == 2:
            gpus = [i for i in range(int(_gpus[0]), int(_gpus[1]))]
        else:
            _gpus = gpus.split(',')
            gpus = [int(x) for x in _gpus]
        return gpus
    except:
        print('the gpus index is error!!! please specify right gpus index, or default use cpu')
        gpus = []
        return gpus


# ===================================== Json Objects =====================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_dict(fname, d, mode='w'):
    # 持久化写入
    with open(fname, mode) as fp:
        # json.dump(d, fp, cls=NpEncoder, indent=1, separators=(',', ': '))
        json.dump(d, fp, cls=NpEncoder)


def load_dict(fname):
    with open(fname, "r") as fp:
        o = json.load(fp, )
        return o


# ===================================== Logger Objects =====================================
def get_logger(name='root'):
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


logger = get_logger('root')


class Metrics:

    def __init__(self, watch, **names):
        self.watch = watch
        self.metric = dict(**names)
        self.metric[watch] = dict(val=0, sum=0, cnt=0, avg=0)
        self.metric['time'] = dict(last=time.time(), val=0, sum=0, cnt=0, avg=0)
        self.metrics = []

    def update(self, **kwargs):
        d = dict(**kwargs)
        for k, v in d.items():
            if k == self.watch:
                self.metric[k]['val'] = float(v)
                self.metric[k]['sum'] += self.metric[k]['val']
                self.metric[k]['cnt'] += 1
                self.metric[k]['avg'] = self.metric[k]['sum'] / self.metric[k]['cnt']
                last = self.metric['time']['last']
                self.metric['time']['last'] = time.time()
                self.metric['time']['val'] = self.metric['time']['last'] - last
                self.metric['time']['sum'] += self.metric['time']['val']
                self.metric['time']['cnt'] += 1
                self.metric['time']['avg'] = self.metric['time']['sum'] / self.metric['time']['cnt']
                with open(self.metric['log_path'], 'a+') as fp:
                    line = json.dumps(self.metric)
                    fp.write(line + '\n')
            else:
                self.metric[k] = v

    def str(self):
        ets = ((self.metric['total_epoch'] - self.metric['epoch']) * self.metric['total_iter'] \
               - self.metric['iter']) * self.metric['time']['avg']
        ets = get_et_time(ets)
        msg = 'Epoch [{}/{}], iter [{}/{}], eta {}, lr {:.6f}, {} {:.4f}({:.4f})'.format(
            self.metric['epoch'], self.metric['total_epoch'],
            self.metric['iter'], self.metric['total_iter'],
            ets, self.metric['lr'], self.watch,
            self.metric[self.watch]['val'], self.metric[self.watch]['avg']
        )
        return msg

import importlib
import os
import json
import numpy as np
import math
import sys
import datetime
import logging
import time
import torch
import torch.distributed as dist
from collections import defaultdict, deque


# ===================================== Common Functions =====================================
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


def check_input(imgs, targets):
    import cv2 as cv
    import numpy as np
    for idx, target in enumerate(targets):
        img = imgs[idx]
        img = img.permute(1, 2, 0)  # C x H x W --> H x W x C
        image = np.array(img.cpu()).copy()
        image = np.array(image[..., ::-1])  # RGB --> BGR
        for box in target['boxes']:
            b = [int(b) for b in box]
            cv.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cv.imshow("img", image)
        cv.waitKey(0)


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


def save_dict(fname, d, mode='w', **kwargs):
    # 持久化写入
    with open(fname, mode) as fp:
        # json.dump(d, fp, cls=NpEncoder, indent=1, separators=(',', ': '))
        json.dump(d, fp, cls=NpEncoder, **kwargs)


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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=", ", log_path='.log.txt', data_path='.data.json', **kwargs):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_path = log_path
        self.data_path = data_path
        self.extras = dict(**kwargs)

    def set_extra(self, **kwargs):
        for k, v in kwargs.items():
            self.extras[k] = v

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq=None, header=None):
        if print_freq is None:
            print_freq = self.extras['print_freq']
        if header is None:
            fmt = ':' + str(len(str(self.extras['total_epoch']))) + 'd'
            fmt_str = 'Epoch [{0' + fmt + '}/{1}]'
            header = fmt_str.format(self.extras['epoch'], self.extras['total_epoch'])

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{total_iter}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{total_iter}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for step, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if step % print_freq == 0 or step == len(iterable) - 1:
                epoch_cnt = self.extras['total_epoch'] + 1 - self.extras['epoch']
                eta_seconds = iter_time.global_avg * (len(iterable) * epoch_cnt - step)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    log_line = log_msg.format(
                        step, total_iter=len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                else:
                    log_line = log_msg.format(
                        step, total_iter=len(iterable), eta=eta_string,
                        meters=str(self), time=str(iter_time), data=str(data_time))

                logger.info(log_line)
                with open(self.log_path, 'a+') as fp:
                    dt = datetime.datetime.now()
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S') + ' [INFO]: '
                    fp.write(datetime_str + log_line + '\n')

                with open(self.data_path, 'a+') as fp:
                    data = dict(
                        epoch=self.extras['epoch'], total_epoch=self.extras['total_epoch'],
                        iter=step, total_iter=len(iterable),
                        iter_time=iter_time.value, data_time=data_time.value
                    )
                    for k, v in self.meters.items():
                        data[k] = v.value
                    data_line = json.dumps(data)
                    fp.write(data_line + '\n')

            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log_line = '{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable))
        logger.info(log_line)
        with open(self.log_path, 'a+') as fp:
            fp.write(log_line + '\n')

# class Metrics:
#
#     def __init__(self, watch, **names):
#         self.watch = watch
#         self.metric = dict(**names)
#         self.metric[watch] = dict(val=0, sum=0, cnt=0, avg=0)
#         self.metric['time'] = dict(last=time.time(), val=0, sum=0, cnt=0, avg=0)
#         self.metrics = []
#
#     def update(self, **kwargs):
#         d = dict(**kwargs)
#         for k, v in d.items():
#             if k == self.watch:
#                 self.metric[k]['val'] = float(v)
#                 self.metric[k]['sum'] += self.metric[k]['val']
#                 self.metric[k]['cnt'] += 1
#                 self.metric[k]['avg'] = self.metric[k]['sum'] / self.metric[k]['cnt']
#                 last = self.metric['time']['last']
#                 self.metric['time']['last'] = time.time()
#                 self.metric['time']['val'] = self.metric['time']['last'] - last
#                 self.metric['time']['sum'] += self.metric['time']['val']
#                 self.metric['time']['cnt'] += 1
#                 self.metric['time']['avg'] = self.metric['time']['sum'] / self.metric['time']['cnt']
#                 with open(self.metric['log_path'], 'a+') as fp:
#                     line = json.dumps(self.metric)
#                     fp.write(line + '\n')
#             else:
#                 self.metric[k] = v
#
#     def str(self):
#         ets = ((self.metric['total_epoch'] - self.metric['epoch']) * self.metric['total_iter'] \
#                - self.metric['iter']) * self.metric['time']['avg']
#         ets = get_et_time(ets)
#         msg = 'Epoch [{}/{}], iter [{}/{}], eta {}, lr {:.4f}, {} {:.4f}({:.4f})'.format(
#             self.metric['epoch'], self.metric['total_epoch'],
#             self.metric['iter'], self.metric['total_iter'],
#             ets, self.metric['lr'], self.watch,
#             self.metric[self.watch]['val'], self.metric[self.watch]['avg']
#         )
#         return msg

import torch
from torchvision.models import resnet34, resnet101
import argparse
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchnet import meter
from sklearn.metrics.classification import classification_report
from torch.utils.data import *
from .build_network import *
from .utils import *
from .data_reader import DataReader, collate_fn


def eval(model, cfg, mode='val', cuda=True):
    data_info = cfg.dataset[mode]
    data_reader = DataReader(
        ann_files=[data_info['ann_file']], img_dirs=[data_info['img_prefix']], transform=None, mode='val',
        img_scale=data_info['img_scale'], keep_ratio=data_info['keep_ratio'],
    )
    data_loader = DataLoader(data_reader, collate_fn=collate_fn, **cfg.val_data_loader)
    y_true, y_pred = [], []
    model.eval()
    for step, (data, target) in tqdm(enumerate(data_loader)):
        # inputs = torch.stack(data)
        # target = torch.from_numpy(np.array(target)).type(torch.LongTensor)
        inputs = data
        targets = target
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        outs = nn.functional.softmax(outputs, dim=1)
        pred = torch.argmax(outs, dim=1)
        y_true.extend(list(targets.cpu().detach().numpy()))
        y_pred.extend(list(pred.cpu().detach().numpy()))
    model.train()
    return classification_report(y_true, y_pred, output_dict=True), \
           classification_report(y_true, y_pred, output_dict=False)


def train_one_epoch(model, cfg, optimizer, lr_scheduler, loss_func, loss_metric, cuda=True):
    ann_files, img_dirs = [], []
    data_info = cfg.dataset[cfg.train_mode[0]]
    for mode in cfg.train_mode:
        data_info = cfg.dataset[mode]
        ann_files.append(data_info['ann_file'])
        img_dirs.append(data_info['img_prefix'])
    data_reader = DataReader(
        ann_files=ann_files, img_dirs=img_dirs, transform=None, mode='train',
        img_scale=data_info['img_scale'], keep_ratio=data_info['keep_ratio'],
    )

    data_loader = DataLoader(data_reader, collate_fn=collate_fn, **cfg.data_loader)
    loss_metric.update(total_iter=len(data_loader))
    model.train()
    for step, (data, target) in enumerate(data_loader):
        # inputs = torch.stack(data)
        # targets = torch.from_numpy(np.array(target)).type(torch.LongTensor)
        inputs = data
        targets = target
        if cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        if cfg.mix['type'] == 'mixup':
            alpha = cfg.mix['alpha']
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs.size(0)).cuda()
            inputs = lam * inputs + (1 - lam) * inputs[index, :]
            targets_a, targets_b = targets, targets[index]
            outputs = model(inputs)
            loss = lam * loss_func(outputs, targets_a) + (1 - lam) * loss_func(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_metric.update(iter=step, loss=loss)
        if step % cfg.freq_cfg['log_print'] == 0 or step == len(data_loader):
            line = loss_metric.str()
            logger.info(line)
            with open(os.path.join(cfg.work_dir, cfg.log['out_file']), 'a+') as fp:
                fp.write(line + '\n')

        # lr_scheduler.step()
        # lr = lr_scheduler.optimizer.param_groups[0]['lr']
        # loss_metric.update(lr=lr)


def train(model, optimizer, lr_scheduler, last_epoch, cfg):
    loss_metric = Metrics(watch='loss', total_epoch=cfg.total_epochs,
                          log_path=os.path.join(cfg.work_dir, cfg.log['data_file']))
    loss_func = build_loss(cfg.loss['type'], **cfg.loss[cfg.loss['type']])
    for epoch in range(last_epoch + 1, cfg.total_epochs):

        lr = lr_scheduler.optimizer.param_groups[0]['lr']
        loss_metric.update(epoch=epoch, lr=lr)
        train_one_epoch(model, cfg, optimizer, lr_scheduler, loss_func, loss_metric)
        if (epoch % cfg.freq_cfg['checkpoint_save'] == 0) or (epoch == cfg.total_epochs - 1):
            mkdirs(cfg.work_dir)
            save_model(
                dict(
                    model=model.state_dict(), optimizer=optimizer,
                    lr_scheduler=lr_scheduler, state=dict(last_epoch=epoch)
                ), epoch, save_dir=cfg.work_dir
            )

        lr_scheduler.step()
        for mode in cfg.val_mode:
            data_info, print_info = eval(model, cfg, mode=mode)
            with open(os.path.join(cfg.work_dir, cfg.log['out_file']), 'a+') as fp:
                fp.write('{}:\n{}\n{}\n'.format(mode, data_info, print_info))
            logger.info('{}:\n{}\n'.format(mode, print_info))

        # loss_meter = meter.AverageValueMeter()
        # confusion_matrix = meter.ConfusionMeter(9)
        # loss_meter.reset()
        # confusion_matrix.reset()
        # meters update
        # loss_meter.add(loss.item())
        # confusion_matrix.add(outs.data, target.data)
        #
        # recalls, precisions, f1_scores = evaluate_confusion_matrix(confusion_matrix)


def test(cfg, epochs):
    if isinstance(cfg, str):
        cfg = import_module(cfg)
    mkdirs(cfg.work_dir)

    cfg.gpus = prase_gpus(cfg.gpus)
    model = build_network(**cfg.model_config, gpus=cfg.gpus)

    logger.info("start test...")
    for epoch in epochs:
        cfg.resume_from = os.path.join(cfg.work_dir, 'epoch_{:06d}.pth'.format(epoch))
        model, optimizer, lr_scheduler, last_epoch = resume_network(model, cfg)
        for mode in cfg.val_mode:
            data_info, print_info = eval(model, cfg, mode=mode)
            with open(os.path.join(cfg.work_dir, cfg.log['out_file']), 'a+') as fp:
                fp.write('{}:\n{}\n{}\n'.format(mode, data_info, print_info))
            logger.info('{}:\n{}\n'.format(mode, print_info))
    logger.info('test successfully!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(cfg):
    setup_seed(666)
    if isinstance(cfg, str):
        cfg = import_module(cfg)
    mkdirs(cfg.work_dir)

    cfg.gpus = prase_gpus(cfg.gpus)
    model = build_network(**cfg.model_config, gpus=cfg.gpus)
    model, optimizer, lr_scheduler, last_epoch = resume_network(model, cfg)

    logger.info("start training...")
    train(model, optimizer, lr_scheduler, last_epoch, cfg)
    logger.info('train successfully!')


if __name__ == '__main__':
    pcfg = import_module("cfg.py")
    dcfg = import_module(pcfg.dataset_cfg_path)
    main(dcfg)

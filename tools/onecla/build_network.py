import torch
import torchvision.models
import torch
import torch.nn as nn
import os
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from .utils import *
import torchvision


def build_optimizer(model, cfg):
    # for param in model.parameters():
    #     param.requires_grad = True

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    if cfg.optimizer['type'] == 'Adam':
        optimizer = torch.optim.Adam(trainable_vars, **cfg.optimizer['Adam'])
    else:
        optimizer = torch.optim.SGD(trainable_vars, **cfg.optimizer['SGD'])

    if cfg.lr_scheduler['type'] == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, **cfg.lr_scheduler['CosineAnnealingLR'])
    else:
        lr_scheduler = ReduceLROnPlateau(optimizer, **cfg.lr_scheduler['ReduceLROnPlateau'])

    return optimizer, lr_scheduler


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, labels):
        classification = nn.functional.softmax(x, dim=1)
        targets = torch.zeros(classification.shape)
        targets = targets.cuda()
        for label, target in zip(labels, targets):
            target[label] = 1

        alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)

        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = -(targets * torch.log(classification + 1e-38) + (1.0 - targets) * torch.log(1.0 - classification + 1e-38))
        cls_loss = focal_weight * bce
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        return cls_loss.sum() / cls_loss.shape[0]


class InverseLoss(nn.Module):

    def __init__(self, alpha=1, beta=0.01):
        super(InverseLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, labels):
        classification = nn.functional.softmax(x, dim=1)
        targets = torch.zeros(classification.shape)
        targets = targets.cuda()
        for label, target in zip(labels, targets):
            target[label] = 1

        # alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
        # alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)

        inverse_weight = torch.where(torch.eq(targets, 1.), classification, 1. - classification)
        inverse_weight = self.alpha / (inverse_weight + self.beta)

        bce = -(targets * torch.log(classification + 1e-38) + (1.0 - targets) * torch.log(1.0 - classification + 1e-38))
        cls_loss = inverse_weight * bce
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        return cls_loss.sum() / cls_loss.shape[0]


def build_loss(type='CrossEntropyLoss', **kwargs):
    if type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**kwargs)
    elif type == 'FocalLoss':
        return FocalLoss(**kwargs)
    elif type == 'InverseLoss':
        return InverseLoss(**kwargs)
    return nn.CrossEntropyLoss(**kwargs)


def resume_network(model, cfg):
    last_epoch = -1
    if cfg.resume_from is not None and (os.path.islink(cfg.resume_from) or os.path.exists(cfg.resume_from)):
        resume_from = cfg.resume_from
        if os.path.islink(resume_from):
            resume_from = os.readlink(resume_from)
        checkpoint = torch.load(resume_from, map_location='cpu')
        optimizer, lr_scheduler = checkpoint['optimizer'], checkpoint['lr_scheduler']
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint['state']['last_epoch']
        logger.info("resume from {}".format(resume_from))
    else:
        optimizer, lr_scheduler = build_optimizer(model, cfg)
        logger.info("warning... resume from {} failed!".format(cfg.resume_from))

    model.train()
    return model, optimizer, lr_scheduler, last_epoch


def build_network(type, name, num_classes, gpus=[], **kwargs):
    model_type = import_module("torchvision.models.{}".format(type))
    model = model_type.__dict__[name](**kwargs)
    obj_list = dir(model)
    if 'fc' in obj_list:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'classifier' in obj_list:
        model.classifier = nn.Sequential(nn.Dropout(p=0.8, inplace=True),
                                         nn.Linear(1280, num_classes))
    # model = FocalModel(model)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(gpus[0])
        model_gpu = nn.DataParallel(module=model, device_ids=gpus)
        model = model_gpu.cuda(gpus[0])
    elif torch.cuda.is_available():
        model.cuda()
    return model


if __name__ == '__main__':
    model = build_network(type='resnet', name='resnet50', num_classes=100, pretrained=False)
    print(model)

# model settings
model_config = dict(
    type='resnet',
    name='resnet50',
    num_classes=21,
    pretrained=True
)

# split data settings
data_name = 'coco_fabric'
data_root = '../../data/classification/coco_fabric/'
img_root = r'C:\Users\zl\liphone\home\data\detection\fabric_defects\coco_fabric\plain_fabric/'
img_save_dir = data_root + 'imgs/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
dataset = dict(
    raw_train_path=data_root + 'train_label.csv',
    raw_test_path='',
    balance=True,
    train=[
        dict(
            name='train',
            ratio=0.6,
            ann_file=data_root + 'annotations/train.csv',
            img_prefix=img_root + 'train/',
            img_scale=(1333, 800),
            img_norm_cfg=img_norm_cfg,
        ),
        dict(
            name='valA',
            ratio=0.2,
            ann_file=data_root + 'annotations/valA.csv',
            img_prefix=img_root + 'train/',
            img_scale=(1333, 800),
            img_norm_cfg=img_norm_cfg,
        ),
        dict(
            name='valB',
            ratio=0.2,
            ann_file=data_root + 'annotations/valB.csv',
            img_prefix=img_root + 'train/',
            img_scale=(1333, 800),
            img_norm_cfg=img_norm_cfg,
        ),
    ],
    test=[
        dict(
            name='test',
            ratio=1,
            ann_file='',
            img_prefix=r'D:\Lance\FTP\large_files\bolu\train',
            img_scale=(1333, 800),
            img_norm_cfg=img_norm_cfg,
        ),
    ],
)

# log settings
log = dict(
    out_file='train_log_out.txt',
    data_file='train_log_data.json'
)

# train process settings
train_mode = ['train','valA', ]
val_mode = ['valB']
total_epochs = 60
work_dir = './work_dirs/' + data_name + '/' + model_config['name']
resume_from = work_dir + '/latest.pth'
load_from = None
mix = dict(
    type='none',
    alpha=2.0,
)
optimizer = dict(
    # type='SGD',
    type='Adam',
    Adam=dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False),
)
lr_scheduler = dict(
    type='CosineAnnealingLR',
    CosineAnnealingLR=dict(T_max=total_epochs),
)
loss = dict(
    type='InverseLoss',
    CrossEntropyLoss=dict(),
    FocalLoss=dict(),
    InverseLoss=dict(),
)
freq_cfg = dict(
    checkpoint_save=1,
    log_print=1,
)
gpus = '0'
data_loader = dict(
    batch_size=84, shuffle=True,
)
val_data_loader = dict(
    batch_size=16, shuffle=False,
)

# model settings
model_config = dict(
    type='resnet',
    name='resnet50',
    num_classes=2,
    pretrained=True
)

# split data settings
data_name = 'coco_alcohol'
data_root = "/home/liphone/undone-work/data/detection/alcohol"
img_save_dir = data_root + "/imgs/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
dataset = dict(
    raw_train_path=data_root + '/annotations/instance_train_alcohol.csv',
    raw_test_path=data_root + '/annotations/instance_train_alcohol.csv',
    balance=False,
    train=dict(
        name='train',
        ann_file=data_root + '/annotations/instance_train_alcohol.csv',
        img_prefix=data_root + '/trainval/',
        img_scale=(224, 224),
        keep_ratio=False,
        img_norm_cfg=img_norm_cfg,
    ),
    val=dict(
        name='val',
        ann_file=data_root + '/annotations/instance_train_alcohol.csv',
        img_prefix=data_root + '/trainval/',
        img_scale=(224, 224),
        keep_ratio=False,
        img_norm_cfg=img_norm_cfg,
    ),
    test=dict(
        name='test',
        ann_file=data_root + '/annotations/instance_test_alcohol.csv',
        img_prefix=data_root + '/trainval/',
        img_scale=(224, 224),
        keep_ratio=False,
        img_norm_cfg=img_norm_cfg,
    ),
)

# log settings
log = dict(
    out_file='train_log_out.txt',
    data_file='train_log_data.json'
)

# train process settings
train_mode = ['train', ]
val_mode = ['val', 'test']
total_epochs = 13 * 4
work_dir = './work_dirs/' + data_name + '/' + model_config['name']
resume_from = work_dir + '/latest.pth'
load_from = None
mix = dict(
    type='none',
    alpha=2.0,
)
optimizer = dict(
    type='SGD',
    # type='Adam',
    Adam=dict(lr=0.0025, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False),
    SGD=dict(lr=0.0025, momentum=0, dampening=0, weight_decay=0, nesterov=False)
)
lr_scheduler = dict(
    type='CosineAnnealingLR',
    CosineAnnealingLR=dict(T_max=total_epochs),
)
loss = dict(
    type='CrossEntropyLoss',
    CrossEntropyLoss=dict(),
    FocalLoss=dict(),
    InverseLoss=dict(alpha=1, beta=0.01),
)
freq_cfg = dict(
    checkpoint_save=1,
    log_print=10,
)
gpus = '0'
data_loader = dict(
    batch_size=2, shuffle=True,
)
val_data_loader = dict(
    batch_size=8, shuffle=False,
)

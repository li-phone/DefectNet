import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_soft_nms():
    from coco_analyze import save_plt
    iou_thr = np.linspace(0.1, 0.9, 9)
    acc = [0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877, 0.877]
    mAP = [0.163, 0.163, 0.163, 0.163, 0.163, 0.163, 0.160, 0.155, 0.135, ]
    AP50 = [0.314, 0.314, 0.314, 0.315, 0.314, 0.311, 0.302, 0.283, 0.241]
    data = pd.DataFrame(data={
        'acc': acc, 'iou_thr': iou_thr, 'mAP': mAP, 'AP@0.50': AP50})

    ax = data.plot.line(
        x='iou_thr', y='mAP', marker='^',
        grid=True,
        xlim=(0.1, 0.9),
        # ylim=(0., 1.),
    )
    plt.ylabel('mAP')
    save_plt('../results/fabric/fabric_defect_detection/soft-nms/soft-nms.jpg')
    plt.show()


def main():
    from batch_train import BatchTrain
    # garbage = BatchTrain(cfg_path='../configs/garbage/garbage_cas_r50_1x.py', data_mode='val',
    #                      train_sleep_time=0, test_sleep_time=-60)
    # garbage.no_trick_train()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    underwater = BatchTrain(cfg_path='../configs/underwater/cas_rcnn_x101_64x4d_fpn_1x.py', data_mode='val',
                            train_sleep_time=0, test_sleep_time=-1)
    underwater.no_trick_train()

    # garbage_train = BatchTrain(cfg_path='../configs/garbage/garbage_cas_rcnn_x101_64x4d_fpn_1x.py', data_mode='val')
    # garbage_train.no_trick_train()

    # underwater = BatchTrain(cfg_path='../configs/underwater/underwater_htc_without_semantic_x101_64x4d_fpn_1x.py',
    #                            data_mode='val')
    # underwater.no_trick_train()

    # fabric_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/fabric.py', data_mode='test')
    # fabric_train.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)], multiscale_mode='value')
    # fabric_train.anchor_cluster_train(anchor_ratios=[0.12, 1.0, 4.43])
    # fabric_train.anchor_cluster_train(anchor_ratios=[0.04, 0.28, 1.0, 4.43, 8.77])
    # fabric_train.anchor_cluster_train(anchor_ratios=[0.04, 0.14, 0.32, 0.69, 1.0, 4.77, 8.84])
    # fabric_train.anchor_cluster_train(anchor_ratios=[0.04, 0.14, 0.27, 0.32, 0.69, 1.0, 3.0, 5.71, 10.22])
    # batrian.baseline_train()
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)], ratio_range=[0.5, 1.5])
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2), (1333, 800)])
    # batrian.multi_scale_train(img_scale=[(2446 / 2, 1000 / 2)])
    # batrian.multi_scale_train(img_scale=[(2446, 1000)])

    # aquatic_train = BatchTrain(cfg_path='../config_alcohol/cascade_rcnn_r50_fpn_1x/aquatic.py', data_mode='val')
    # aquatic_train.compete_train()


if __name__ == '__main__':
    main()

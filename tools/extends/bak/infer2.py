# %%
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import json
import argparse
import os
import glob
from tqdm import tqdm
import time
from pandas.io.json import json_normalize
import pandas as pd
from mmdet.core import coco_eval
from pycocotools.coco import COCO
import numpy as np
from sklearn.metrics import classification_report
from utilities.draw_util import draw_coco


def save_json(results, submit_filename):
    with open(submit_filename, 'w') as fp:
        json.dump(results, fp, indent=4, separators=(',', ': '))


def infer_by_path(model, image_paths):
    results = dict(images=[], annotations=[])
    # name2label = {1: 1, 9: 2, 5: 3, 3: 4, 4: 5, 0: 6, 2: 7, 8: 8, 6: 9, 10: 10, 7: 11}
    # label2name = {v: k for k, v in name2label.items()}
    fps_times = []
    for img_id, path in tqdm(enumerate(image_paths)):
        results['images'].append(dict(file_name=os.path.basename(path), id=img_id))
        start_t = time.time()
        result = inference_detector(model, path)
        fps_times.append(time.time() - start_t)
        for idx, pred in enumerate(result):
            # category_id = label2name[idx+1]
            category_id = idx + 1
            for x in pred:
                bbox_pred = {
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])],
                    "score": float(x[4]),
                }
                results['annotations'].append(bbox_pred)
    return results, fps_times


def infer_by_ann(model, img_dir, anns):
    results = dict(images=[], annotations=[])
    # name2label = {1: 1, 9: 2, 5: 3, 3: 4, 4: 5, 0: 6, 2: 7, 8: 8, 6: 9, 10: 10, 7: 11}
    # label2name = {v: k for k, v in name2label.items()}
    fps_times = []
    for i, ann in tqdm(enumerate(anns.dataset['images'])):
        img_id = ann['id']
        path = os.path.join(img_dir, ann['file_name'])
        results['images'].append(dict(file_name=os.path.basename(path), id=img_id))
        start_t = time.time()
        result = inference_detector(model, path)
        fps_times.append(time.time() - start_t)
        if isinstance(result, list):
            for idx, pred in enumerate(result):
                # category_id = label2name[idx+1]
                category_id = idx
                for x in pred:
                    bbox_pred = {
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])],
                        "score": float(x[4]),
                    }
                    results['annotations'].append(bbox_pred)
        elif isinstance(result, int):
            pass

    return results, fps_times


def have_defect(anns, threshold=0.05):
    annotations = json_normalize(anns['annotations'])
    det_results = []
    for image in anns['images']:
        defect_num = 0
        if annotations.shape[0] > 0:
            ann = annotations[annotations['image_id'] == image['id']]
            for j in range(ann.shape[0]):
                a = ann.iloc[j]
                if 'score' in a:
                    if a['score'] > threshold and a['category_id'] > 0:
                        defect_num += 1
                else:
                    if a['category_id'] > 0:
                        defect_num += 1
        if 0 == defect_num:
            det_results.append(0)
        else:
            det_results.append(1)
    assert len(det_results) == len(anns['images'])
    return det_results


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config',
        default='../config_alcohol/cascade_rcnn_r50_fpn_1x/DefectNet_no_bg.py',
        help='train config file path')
    parser.add_argument(
        '--resume_from',
        default='../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/DefectNet_have_bg/epoch_12.pth',
        help='train config file path')
    parser.add_argument(
        '--ann_file',
        default='/home/liphone/undone-work/data/detection/alcohol/annotations/instance_test_alcohol.json')
    parser.add_argument(
        '--img_dir',
        default='/home/liphone/undone-work/data/detection/alcohol/trainval')
    parser.add_argument(
        '--work_dir',
        default='../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/DefectNet_have_bg/',
        help='train config file path')
    args = parser.parse_args()

    return args


def draw(img_dir, work_dir, ann_file):
    defect_name2label = {
        0: '背景',
        1: '瓶盖破损',
        2: '瓶盖变形',
        3: '瓶盖坏边',
        4: '瓶盖打旋',
        5: '瓶盖断点',
        6: '标贴歪斜',
        7: '标贴起皱',
        8: '标贴气泡',
        9: '喷码正常',
        10: '喷码异常'
    }
    label_list = [v for k, v in defect_name2label.items()]
    draw_coco(
        ann_file,
        img_dir,
        os.path.join(work_dir, '.infer_tmp'),
        label_list,
    )


def infer_main(config, resume_from, ann_file, img_dir, work_dir):
    model_name = os.path.basename(config).split('.')[0]
    last_epoch = os.path.basename(resume_from).split('.')[0]

    model = init_detector(config, resume_from, device='cuda:0')

    # get test ann file
    anns = COCO(ann_file)
    anns.dataset['images'] = anns.dataset['images']
    results, fps = infer_by_ann(model, img_dir, anns)

    # save the defect images results
    filename = '{}_{}_defect_image_cascade_rcnn_r50_fpn_1x_.json'.format(model_name, last_epoch)
    submit_filename = os.path.join(work_dir, filename)
    save_json(results['annotations'], submit_filename)
    print('infer all test images ok!')

    # coco eval for defect images
    defect_rpt = coco_eval(submit_filename, ['bbox'], ann_file, classwise=True)
    mAP = defect_rpt[0][0] + '\n' + defect_rpt[0][1]

    y_pred = have_defect(results)
    y_true = have_defect(anns.dataset)
    acc = classification_report(y_true, y_pred, output_dict=False)
    defect_fps = [fps[i] for i, x in enumerate(y_true) if x != 0]
    normal_fps = [fps[i] for i, x in enumerate(y_true) if x == 0]
    assert len(defect_fps) + len(normal_fps) == len(fps)
    return dict(acc=acc, mAP=mAP, fps=np.mean(fps), fps_defect=np.mean(defect_fps), fps_no_defect=np.mean(normal_fps))


def main():
    args = parse_args()
    rpt = infer_main(args.config, args.resume_from, args.ann_file, args.img_dir, args.work_dir)
    print(rpt)


if __name__ == '__main__':
    main()

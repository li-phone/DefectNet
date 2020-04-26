from pycocotools.coco import COCO
import pandas as pd
import os
import json
from pandas import json_normalize


def have_defect(anns, images, threshold=0.05, background_id=0):
    assert background_id == 0
    if isinstance(anns, str):
        with open(anns) as fp:
            anns = json.load(fp)
    if isinstance(anns, dict):
        anns = anns['annotations']
    assert isinstance(anns, list)
    annotations = json_normalize(anns)
    # assert annotations.shape[0] == annotations[annotations['score'] > 0.05].shape[0]
    det_results = []
    for image in images:
        defect_num = 0
        if annotations.shape[0] > 0:
            ann = annotations[annotations['image_id'] == image['id']]
            for j in range(ann.shape[0]):
                a = ann.iloc[j]
                if 'score' in a:
                    if a['score'] > threshold and a['category_id'] != background_id:
                        defect_num += 1
                else:
                    if a['category_id'] != background_id:
                        defect_num += 1
        det_results.append(defect_num)
    assert len(det_results) == len(images)
    return det_results


def coco_defect2csv(ann_path, save_name):
    coco = COCO(ann_path)
    dataset = coco.dataset

    true_nums = have_defect(dataset, dataset['images'])
    y_true = [0 if x == 0 else 1 for x in true_nums]
    ids = [r['file_name'] for r in dataset['images']]
    data = pd.DataFrame({'id': ids, 'label': y_true})
    data.to_csv(save_name, index=False)


if __name__ == '__main__':
    coco_defect2csv(
        '../../work_dirs/data/bottle/annotations/train.json',
        '../../work_dirs/data/bottle/annotations/cls_train.csv'
    )
    coco_defect2csv(
        '../../work_dirs/data/bottle/annotations/test.json',
        '../../work_dirs/data/bottle/annotations/cls_test.csv'
    )

    coco_defect2csv(
        '../../work_dirs/data/fabric/annotations/train.json',
        '../../work_dirs/data/fabric/annotations/cls_train.csv'
    )
    coco_defect2csv(
        '../../work_dirs/data/fabric/annotations/test.json',
        '../../work_dirs/data/fabric/annotations/cls_test.csv'
    )

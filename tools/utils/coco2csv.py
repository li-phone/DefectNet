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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Check ann_file')
    parser.add_argument('ann_file', help='coco annotation file')
    parser.add_argument('save_name', help='save_name for defect csv file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_defect2csv(args.ann_file, args.save_name)


if __name__ == '__main__':
    main()

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize


def count_coco(coco, min_num=200):
    if isinstance(coco, str):
        coco = COCO(coco)
    dataset = coco.dataset
    anns = json_normalize(dataset['annotations'])
    label_names = [v['name'] for v in dataset['categories']]
    results = anns.groupby('category_id').count()
    results['name'] = label_names
    results.sort_values(by="image_id", ascending=False, inplace=True)
    results = results[results['id'] >= min_num]
    results = results.to_dict("records")
    return results


def filter_coco(coco, label_names):
    if isinstance(coco, str):
        coco = COCO(coco)
    dataset = coco.dataset
    name2id = {v['name']: v['id'] for v in dataset['categories']}
    keep_ids = {c: name2id[c] for c in label_names}
    # 把不在label_names内的标注删除
    dataset['annotations'] = [v for v in dataset['annotations'] if v['category_id'] in keep_ids.values()]
    anns = json_normalize(dataset['annotations'])
    images = []
    for image in tqdm(dataset['images']):
        keep = anns[anns['image_id'] == image['id']]
        if len(keep) != 0:
            images.append(image)
    dataset['images'] = images
    dataset['categories'] = [v for v in dataset['categories'] if v['id'] in keep_ids.values()]
    return dataset


def reorder_coco(dataset):
    img_id2new_id = {}
    img_id = 0
    for image in dataset['images']:
        img_id2new_id[image['id']] = img_id
        img_id += 1
    for image in dataset['images']:
        image['id'] = img_id2new_id[image['id']]
    ann_id2new_id = {}
    cat_id = 0
    for cat in dataset['categories']:
        ann_id2new_id[cat['id']] = cat_id
        cat_id += 1
    for cat in dataset['categories']:
        cat['id'] = ann_id2new_id[cat['id']]
    ann_id = 0
    for ann in dataset['annotations']:
        ann['id'] = ann_id
        ann_id += 1
        ann['image_id'] = img_id2new_id[ann['image_id']]
        ann['category_id'] = ann_id2new_id[ann['category_id']]
    return dataset


def main():
    # 统计瑕疵的数量，剔除数量少的瑕疵类别
    data_file = None
    data_file = "/home/lifeng/data/detection/fabric/annotations/no_filter_annotations/instance_all.json"
    data_count = count_coco(data_file, min_num=200)
    label_aps = {'背景': 0.0, '破洞': 0.4175422079242009, '水渍': 0.0, '油渍': 0.3279592173503065, '污渍': 0.13599813471911032,
                 '三丝': 0.34710367555092375, '结头': 0.21374970443416194, '花板跳': 0.46009489998633096,
                 '百脚': 0.1590055697682288, '毛粒': 0.05473379174635575, '粗经': 0.07269697472928321,
                 '松经': 0.16737380001863666, '断经': 0.13401281992864658, '吊经': 0.04574514594316574,
                 '粗维': 0.21407319585000775, '纬缩': 0.05642950992549857, '浆斑': 0.3505216168331181,
                 '整经结': 0.1867522752525842, '星跳': 0.062074832492071956, '跳花': 0.23662958328799913,
                 '断氨纶': 0.15111392769687607, '稀密档': 0.014563660243815215, '浪纹档': 0.0, '色差档': 0.0,
                 '磨痕': 0.05148514851485147, '轧痕': 0.155342394380527, '修痕': 0.05486429049804582,
                 '烧毛痕': 0.6311350777934936, '死皱': 0.0012376237623762376, '云织': 0.0, '双纬': 0.0, '双经': 0.0, '跳纱': 0.0,
                 '筘路': 0.0, '纬纱不良': 0.0}
    label_names = {v['name']: v for v in data_count}
    for k, v in label_aps.items():
        if False and v < 0.10 and k != '背景' and k in label_names:
            label_names.pop(k)
    print(len(label_names), label_names)
    # keep this categories
    coco = filter_coco(data_file, label_names.keys())
    print("all: {}, normal: {}, defective: {}".format(len(coco['images']), data_count[0]['id'],
                                                      len(coco['images']) - data_count[0]['id']))
    coco = reorder_coco(coco)
    # save_name = "/home/lifeng/data/detection/fabric/annotations/instance_all.json"
    with open(save_name, "w") as fp:
        json.dump(coco, fp)
    print('process ok!')
    # all: 7802, normal: 3663, defective: 4139, 15
    # all: 8080, normal: 3663, defective: 4417, 17


if __name__ == '__main__':
    main()

import glob
from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET
import os
import json
import pandas as pd
import numpy as np
import argparse
import random

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize


class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def _parse_int_float_bool(self, val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(',')]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def _get_box(points):
    min_x = min_y = np.inf
    max_x = max_y = 0
    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def read_xml(xml_path, img_dir):
    if isinstance(xml_path, str):
        xml_path = glob.glob(os.path.join(xml_path, '*.xml'))
    anns = []
    for path in tqdm(xml_path):
        img_id = os.path.basename(path)
        img_id = img_id.split(".xml")[0]
        img_path = glob.glob(os.path.join(img_dir, img_id + ".*"))
        assert len(img_path) == 1
        img_path = img_path[0]
        if not os.path.exists(img_path):
            print(img_path, "not exists")
        tree = ET.parse(path)
        root = tree.getroot()
        size = root.find('size')
        if size is not None:
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None:
                iscrowd = int(difficult.text)
            else:
                iscrowd = 0
            label_name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xmin, xmax, ymin, ymax = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                                      float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox = [xmin, ymin, xmax, ymax]
            ann = dict(
                file_name=os.path.basename(img_path),
                label=label_name,
                bbox=bbox,
                iscrowd=iscrowd,
            )
            anns.append(ann)
    return json_normalize(anns)


def read_json(anns):
    if isinstance(anns, str):
        if anns[-5:] == '.json':
            with open(anns) as fp:
                anns = json.load(fp)
        elif anns[-4:] == '.csv':
            anns = pd.read_csv(anns)
    if isinstance(anns, list):
        anns = json_normalize(anns)
    assert isinstance(anns, pd.DataFrame)
    if 'bbox' not in list(anns.columns):
        bbox = []
        for i in range(anns.shape[0]):
            r = anns.iloc[i]
            b = [r['xmin'], r['ymin'], r['xmax'], r['ymax']]
            b = [float(_) for _ in b]
            bbox.append(b)
        anns['bbox'] = bbox
    if 'name' in list(anns.columns):
        anns = anns.rename(columns={'name': 'file_name'})
    if 'defect_name' in list(anns.columns):
        anns = anns.rename(columns={'defect_name': 'label'})
    return anns


def convert2coco(anns, save_name, img_dir, label2cat=None, bgcat=None, supercategory=None, info=None,
                 license=None):
    assert isinstance(anns, pd.DataFrame)
    if 'id' not in list(anns.columns):
        anns['id'] = list(range(anns.shape[0]))
    if label2cat is None:
        if 'category' in list(anns.columns):
            anns.rename(columns={'category': 'label'}, inplace=True)
        label2cat = list(np.array(anns['label'].unique()))
        if bgcat in label2cat:
            label2cat.remove(bgcat)
        label2cat = np.sort(label2cat)
        label2cat = list(label2cat)
        label2cat = {i + 1: v for i, v in enumerate(label2cat)}
        # not including background label
        # if bgcat is not None:
        #     label2cat[0] = bgcat

    if supercategory is None:
        supercategory = [None] * (len(label2cat) + 1)

    coco = dict(info=info, license=license, categories=[], images=[], annotations=[])
    label2cat = {v: k for k, v in label2cat.items()}
    coco['categories'] = [dict(name=str(k), id=v, supercategory=supercategory[v]) for k, v in label2cat.items()]

    images = list(anns['file_name'].unique())
    import cv2 as cv
    for i, v in tqdm(enumerate(images)):
        if os.path.exists(os.path.join(img_dir, v)):
            img_ = cv.imread(os.path.join(img_dir, v))
            height_, width_, _ = img_.shape
        else:
            row = anns.iloc[i]
            height_, width_, _ = int(row['image_height']), int(row['image_width']), 3
        assert height_ is not None and width_ is not None
        keep_df = anns[anns['file_name'] == v]
        cats = list(keep_df['label'].unique())
        img_label = 0 if len(cats) == 0 or (len(cats) == 1 and cats[0] == bgcat) else 1
        coco['images'].append(dict(file_name=v, id=i, width=width_, height=height_, img_label=img_label))
    image2id = {v['file_name']: v['id'] for i, v in enumerate(coco['images'])}

    annotations = anns.to_dict('records')
    for v in annotations:
        if v['label'] is None or v['label'] == bgcat or v['bbox'] is None:
            continue
        bbox = v['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        ann = dict(
            id=v['id'],
            image_id=image2id[v['file_name']],
            category_id=label2cat[v['label']],
            bbox=_get_box(points),
            iscrowd=0,
            ignore=0,
            area=area
        )
        coco['annotations'].append(ann)
    save_name = save_name.replace('\\', '/')
    save_dir = save_name[:save_name.rfind('/')]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_name, 'w', encoding='utf-8') as fp:
        json.dump(coco, fp)


def imgdir2coco(coco_sample, save_name, img_dir):
    if isinstance(coco_sample, str):
        from pycocotools.coco import COCO
        coco_sample = COCO(coco_sample)
    coco_sample = coco_sample.dataset
    coco_test = dict(
        info=coco_sample['info'], license=coco_sample['license'], categories=coco_sample['categories'],
        images=[], annotations=[])
    images = glob.glob(os.path.join(img_dir, '*'))

    import cv2 as cv
    for i, v in tqdm(enumerate(images)):
        v = os.path.basename(v)
        if os.path.exists(os.path.join(img_dir, v)):
            img_ = cv.imread(os.path.join(img_dir, v))
            height_, width_, _ = img_.shape
        else:
            height_, width_, _ = None, None, None
        coco_test['images'].append(dict(file_name=v, id=i, width=width_, height=height_))

    with open(save_name, 'w') as fp:
        json.dump(coco_test, fp)


def parse_args():
    parser = argparse.ArgumentParser(description='Transform other dataset format into coco format')
    parser.add_argument('--ann_or_dir',
                        default="/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/train_annos.json",
                        help='annotation file or test image directory')
    parser.add_argument('--save_name',
                        default="/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_all.json",
                        help='save_name')
    parser.add_argument('--img_dir',
                        default="/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/train_imgs/",
                        help='img_dir')
    parser.add_argument(
        '--options',
        nargs='+', action=MultipleKVAction,
        help='bgcat(background label), info=None, license=None'
    )
    parser.add_argument(
        '--fmt',
        choices=['json', 'xml', 'test_dir', 'csv'],
        default='json', help='format type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    kwargs = {} if args.options is None else args.options
    if args.fmt == 'xml':
        anns = read_xml(args.ann_or_dir, args.img_dir)
        convert2coco(anns, args.save_name, args.img_dir, **kwargs)
    elif args.fmt == 'csv' or args.fmt == 'json':
        anns = read_json(args.ann_or_dir)
        convert2coco(anns, args.save_name, args.img_dir, **kwargs)
    elif args.fmt == 'test_dir':
        imgdir2coco(args.ann_or_dir, args.save_name, args.img_dir)


if __name__ == '__main__':
    main()

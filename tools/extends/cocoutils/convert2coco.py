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


# def fabric2coco():
#     ann_file = '/home/liphone/undone-work/data/detection/fabric/annotations/anno_train_20190818-20190928.json'
#     save_ann_name = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train,type=34,.json'
#     img_dir = '/home/liphone/undone-work/data/detection/fabric/trainval'
#     with open(ann_file) as fp:
#         ann_json = json.load(fp)
#     normal_images = glob.glob(os.path.join(img_dir, 'normal_Images_*.jpg'))
#     bbox = [0, 0, 32, 32]
#     for p in normal_images:
#         ann_json.append(dict(name=os.path.basename(p), defect_name='背景', bbox=bbox))
#     label2cat = {
#         '背景': 0, '破洞': 1, '水渍': 2, '油渍': 3, '污渍': 4, '三丝': 5, '结头': 6, '花板跳': 7, '百脚': 8, '毛粒': 9,
#         '粗经': 10, '松经': 11, '断经': 12, '吊经': 13, '粗维': 14, '纬缩': 15, '浆斑': 16, '整经结': 17, '星跳': 18, '跳花': 19,
#         '断氨纶': 20, '稀密档': 21, '浪纹档': 22, '色差档': 23, '磨痕': 24, '轧痕': 25, '修痕': 26, '烧毛痕': 27, '死皱': 28, '云织': 29,
#         '双纬': 30, '双经': 31, '跳纱': 32, '筘路': 33, '纬纱不良': 34,
#     }
#     defect_name2label = {
#         '背景': 0, '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
#         '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
#         '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
#         '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
#     }
#     # name2label_20 = {
#     #     '背景': 0, '破洞': 1, '水渍_油渍_污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
#     #     '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳_跳花': 16,
#     #     '断氨纶': 17, '稀密档_浪纹档_色差档': 18, '磨痕_轧痕_修痕_烧毛痕': 19,
#     #     '死皱_云织_双纬_双经_跳纱_筘路_纬纱不良': 20,
#     # }
#     # transform2coco(ann_json, save_ann_name, label2cat)
#     cn2eng = {
#         '背景': 'background', '破洞': 'hole', '水渍': 'water stain', '油渍': 'oil stain',
#         '污渍': 'soiled', '三丝': 'three silk', '结头': 'knots', '花板跳': 'card skip', '百脚': 'mispick',
#         '毛粒': 'card neps', '粗经': 'coarse end', '松经': 'loose warp', '断经': 'cracked ends',
#         '吊经': 'buttonhold selvage', '粗维': 'coarse picks', '纬缩': 'looped weft', '浆斑': 'hard size',
#         '整经结': 'warping knot', '星跳': 'stitch', '跳花': 'skips',
#         '断氨纶': 'broken spandex', '稀密档': 'thin thick place', '浪纹档': 'buckling place', '色差档': 'color shading',
#         '磨痕': 'smash', '轧痕': 'roll marks', '修痕': 'take marks', '烧毛痕': 'singeing', '死皱': 'crinked',
#         '云织': 'uneven weaving', '双纬': 'double pick', '双经': 'double end', '跳纱': 'felter', '筘路': 'reediness',
#         '纬纱不良': 'bad weft yarn',
#     }
#     label_list = [v for k, v in cn2eng.items()]
#     from draw_util import draw_coco
#     draw_coco(
#         save_ann_name, img_dir, '/home/liphone/undone-work/data/detection/fabric/.instance_train,type=34,', label_list
#     )
#
#
# def underwater2coco():
#     data_root = 'E:/liphone/data/images/detections/underwater'
#
#     save_name = data_root + '/annotations/underwater_train.json'
#     img_dir = data_root + '/train/image'
#     if not os.path.exists(save_name):
#         xml_dir = data_root + '/train/box'
#         anns = xml2list(xml_dir, img_dir)
#         transform2coco(anns, save_name, img_dir=img_dir, bgcat={'id': 0, 'name': 'waterweeds'})
#
#     test_name = data_root + '/annotations/underwater_testB.json'
#     if not os.path.exists(test_name):
#         test_img_dir = data_root + '/test-B-image'
#         imgdir2coco(save_name, test_name, test_img_dir)
#
#     # from draw_util import draw_coco
#     # draw_coco(
#     #     save_name, img_dir, data_root + '/train/.aquatic'
#     # )
#

# def main():
#     underwater2coco()
#
#     # img_dir = '/home/liphone/undone-work/data/detection/fabric/trainval'
#     # cn2eng = {
#     #     '背景': 'background', '破洞': 'hole', '水渍': 'water stain', '油渍': 'oil stain',
#     #     '污渍': 'soiled', '三丝': 'three silk', '结头': 'knots', '花板跳': 'card skip', '百脚': 'mispick',
#     #     '毛粒': 'card neps', '粗经': 'coarse end', '松经': 'loose warp', '断经': 'cracked ends',
#     #     '吊经': 'buttonhold selvage', '粗维': 'coarse picks', '纬缩': 'looped weft', '浆斑': 'hard size',
#     #     '整经结': 'warping knot', '星跳': 'stitch', '跳花': 'skips',
#     #     '断氨纶': 'broken spandex', '稀密档': 'thin thick place', '浪纹档': 'buckling place', '色差档': 'color shading',
#     #     '磨痕': 'smash', '轧痕': 'roll marks', '修痕': 'take marks', '烧毛痕': 'singeing', '死皱': 'crinked',
#     #     '云织': 'uneven weaving', '双纬': 'double pick', '双经': 'double end', '跳纱': 'felter', '筘路': 'reediness',
#     #     '纬纱不良': 'bad weft yarn',
#     # }
#     # label_list = [v for k, v in cn2eng.items()]
#     #
#     # import copy
#     # from draw_util import draw_coco
#     # from utils import load_dict
#     # gt_files = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_test_rate=0.80.json'
#     # gt_results = load_dict(gt_files)
#     #
#     # baseline_path = '/home/liphone/undone-work/defectNet/DefectNet/work_dirs/fabric/cascade_rcnn_r50_fpn_1x/fabric_baseline' + \
#     #                 '/fabric_baseline+mode=baseline/mode=test,.bbox.json'
#     # baseline_boxes = load_dict(baseline_path)
#     # baseline_results = copy.deepcopy(gt_results)
#     # baseline_results['annotations'] = baseline_boxes
#     # draw_coco(
#     #     baseline_results,
#     #     img_dir, '/home/liphone/undone-work/data/detection/fabric/baseline_results+type=34,', label_list,
#     #     thresh=0.05, fontsize=16 * 4
#     # )
#     #
#     # trick_path = '/home/liphone/undone-work/defectNet/DefectNet/work_dirs/fabric/cascade_rcnn_r50_fpn_1x/fabric_baseline' + \
#     #              '/fabric_baseline+baseline+multi-scale+anchor_clusters' + \
#     #              '/baseline+multi-scale+anchor_clusters+soft-nms+mode=test,.bbox.json'
#     # trick_boxes = load_dict(trick_path)
#     # trick_results = copy.deepcopy(gt_results)
#     # trick_results['annotations'] = trick_boxes
#     # draw_coco(
#     #     trick_results,
#     #     img_dir, '/home/liphone/undone-work/data/detection/fabric/baseline_results+tricks+type=34,', label_list,
#     #     thresh=0.05, fontsize=16 * 4
#     # )

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
                        default="/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_train.json",
                        help='annotation file or test image directory')
    parser.add_argument('--save_name',
                        default="/home/lifeng/undone-work/dataset/detection/tile/annotations/submit_testA.json",
                        help='save_name')
    parser.add_argument('--img_dir',
                        default="/home/lifeng/undone-work/dataset/detection/tile/tile_round1_testA_20201231/testA_imgs/", help='img_dir')
    parser.add_argument(
        '--options',
        nargs='+', action=MultipleKVAction,
        help='bgcat(background label), info=None, license=None'
    )
    parser.add_argument(
        '--fmt',
        choices=['json', 'xml', 'test_dir', 'csv'],
        default='test_dir', help='format type')
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

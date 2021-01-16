from pycocotools.coco import COCO
from mmdet.datasets.pipelines.loading import LoadImageFromFile, LoadAnnotations
from mmdet.datasets.pipelines.cut_roi import CutROI
from mmdet.datasets.pipelines.cut_image import CutImage
from tqdm import tqdm
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
from convert2coco import _get_box

loadImage = LoadImageFromFile()
cutROI = CutROI()
cutImage = CutImage(window=(1000, 1000), step=(500, 500), order_index=False, )

img_dir = "/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/train_imgs/"
save_img_dir = "/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/cut_train_imgs/"
ann_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_all.json"
save_ann_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/cut_images_all.json"


def process_cut_images():
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    original_coco = COCO(ann_file)
    dataset = original_coco.dataset
    last_index = -1
    if os.path.exists(save_ann_file):
        save_coco = COCO(save_ann_file)
        last_index = save_coco.dataset['last_index']
        new_images, new_annotations = save_coco.dataset['images'], save_coco.dataset['annotations']
    else:
        new_images, new_annotations = [], []
    for idx, image in tqdm(enumerate(dataset['images'])):
        if idx < last_index:
            continue
        image['filename'] = image['file_name']
        imgIds = image['id']
        annIds = original_coco.getAnnIds(imgIds=imgIds)
        anns = original_coco.loadAnns(annIds)
        bboxes = [x['bbox'] for x in anns]
        bboxes = np.array(bboxes)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        labels = [x['category_id'] for x in anns]
        labels = np.array(labels)
        anns2 = {'bboxes': bboxes, 'labels': labels}
        results = {
            'img_prefix': img_dir,
            'img_info': image, 'ann_info': anns2}
        results = loadImage.__call__(results)
        results = cutROI.__call__(results)
        results = cutImage.__call__(results)
        if results is None: results = []
        for i, result in enumerate(results):
            tmp_image = {k: v for k, v in image.items()}
            tmp_image['file_name'] = "{}_{}.jpg".format(tmp_image['file_name'], i)
            tmp_image['id'] = len(new_images)
            tmp_image['height'] = result['img'].shape[0]
            tmp_image['width'] = result['img'].shape[1]
            new_images.append(tmp_image)
            for bbox, label in zip(result['gt_bboxes'], result['gt_labels']):
                # b = list(map(int, bbox))
                # cv2.rectangle(result['img'], tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 3)
                # cv2.imwrite("/home/lifeng/undone-work/DefectNet/tools/tmp/{}.jpg".format(1), result['img'])
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                ann = dict(
                    id=len(new_annotations),
                    image_id=tmp_image['id'],
                    category_id=int(label),
                    bbox=_get_box(points),
                    iscrowd=0,
                    ignore=0,
                    area=area
                )
                new_annotations.append(ann)
            save_name = os.path.join(save_img_dir, tmp_image['file_name'])
            if not os.path.exists(save_name):
                cv2.imwrite(save_name, result['img'])
        if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset['images']):
            tmp_coco = dict(info=dataset['info'], license=dataset['license'], categories=dataset['categories'],
                            images=new_images, last_index=idx, annotations=new_annotations)
            with open(save_ann_file, "w") as fp:
                json.dump(tmp_coco, fp)
            print("process {}/{}...".format(idx + 1, len(dataset['images'])))
    print('process ok')


process_cut_images()

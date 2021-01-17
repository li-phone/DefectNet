from pycocotools.coco import COCO
from mmdet.datasets.pipelines.loading import LoadImageFromFile, LoadAnnotations
from mmdet.datasets.pipelines.cut_roi import CutROI
from mmdet.datasets.pipelines.cut_image import CutImage
from mmdet.datasets.pipelines import Compose
from tqdm import tqdm
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
import math
import threading
from convert2coco import _get_box


class CutConfig(object):
    # process module
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        # dict(type='CutROI', training=False),
        dict(type='CutImage', training=True, window=(600, 600), step=(300, 300), order_index=False)
    ]
    compose = Compose(train_pipeline)

    # data module
    img_dir = "/home/lifeng/undone-work/dataset/detection/tile/raw/tile_round1_train_20201231/train_imgs/"
    save_img_dir = "/home/lifeng/undone-work/dataset/detection/tile/trainval/cut_600x600/"
    ann_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_all-check.json"
    save_ann_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/cut_600x600/cut_600x600_all.json"
    original_coco = COCO(ann_file)
    main_thread_lock = threading.Lock()
    new_images, new_annotations = [], []
    num_workers = 7
    process_cnt = 0


def do_work(images, config):
    for image in tqdm(images):
        image['filename'] = image['file_name']
        imgIds = image['id']
        annIds = config.original_coco.getAnnIds(imgIds=imgIds)
        anns = config.original_coco.loadAnns(annIds)
        bboxes = [x['bbox'] for x in anns]
        bboxes = np.array(bboxes)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        labels = [x['category_id'] for x in anns]
        labels = np.array(labels)
        anns2 = {'bboxes': bboxes, 'labels': labels}
        results = {
            'img_prefix': config.img_dir,
            'img_info': image, 'ann_info': anns2}
        results = config.compose(results)
        if results is None: results = []
        for i, result in enumerate(results):
            tmp_image = {k: v for k, v in image.items()}
            tmp_image['file_name'] = "{}__{}.jpg".format(tmp_image['file_name'], i)
            tmp_image['height'] = result['img'].shape[0]
            tmp_image['width'] = result['img'].shape[1]
            save_name = os.path.join(config.save_img_dir, tmp_image['file_name'])
            if True or not os.path.exists(save_name):
                cv2.imwrite(save_name, result['img'])
            with config.main_thread_lock:
                tmp_image['id'] = len(config.new_images)
                config.new_images.append(tmp_image)
                for bbox, label in zip(result['gt_bboxes'], result['gt_labels']):
                    # b = list(map(int, bbox))
                    # cv2.rectangle(result['img'], tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 3)
                    # cv2.imwrite("/home/lifeng/undone-work/DefectNet/tools/tmp/{}.jpg".format(1), result['img'])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                    ann = dict(
                        id=len(config.new_annotations),
                        image_id=tmp_image['id'],
                        category_id=int(label),
                        bbox=_get_box(points),
                        iscrowd=0,
                        ignore=0,
                        area=area
                    )
                    config.new_annotations.append(ann)
        config.process_cnt += 1
        if config.process_cnt % 1 == 0 or config.process_cnt == len(images):
            print("process {}/{}...".format(config.process_cnt, len(images)))
    return True


def main():
    config = CutConfig()
    if not os.path.exists(config.save_img_dir):
        os.makedirs(config.save_img_dir)
    if not os.path.exists(os.path.dirname(config.save_ann_file)):
        os.makedirs(os.path.dirname(config.save_ann_file))
    dataset = config.original_coco.dataset
    dataset['images'] = dataset['images']
    per_work_size = len(dataset['images']) // config.num_workers
    fetch, cnt = [], 0
    threads = []
    for i in range(config.num_workers):
        start = i * per_work_size
        end = start + per_work_size
        if (i + 1) == config.num_workers:
            end = len(dataset['images'])
        images = dataset['images'][start:end]
        cnt += len(images)
        threads.append(threading.Thread(target=do_work, args=(images, config)))
    assert cnt == len(dataset['images'])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tmp_coco = dict(info=dataset['info'], license=dataset['license'], categories=dataset['categories'],
                    images=config.new_images, annotations=config.new_annotations)
    with open(config.save_ann_file, "w") as fp:
        json.dump(tmp_coco, fp)


if __name__ == '__main__':
    main()

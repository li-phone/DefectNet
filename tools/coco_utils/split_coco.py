import os
import json
import random
from pycocotools.coco import COCO


def save_dict(fname, d, mode='w', **kwargs):
    # 持久化写入
    with open(fname, mode) as fp:
        # json.dump(d, fp, cls=NpEncoder, indent=1, separators=(',', ': '))
        json.dump(d, fp, **kwargs)


def get_coco_by_imgids(coco, img_ids):
    images = coco.loadImgs(ids=img_ids)
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    annotations = coco.loadAnns(ids=ann_ids)
    categories = coco.dataset['categories']
    instance_coco = dict(info="", images=images, license="", categories=categories, annotations=annotations)
    return instance_coco


def split_coco(ann_path, save_dir, rate=0.8, random_state=666):
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    random.seed(random_state)
    random.shuffle(image_ids)

    train_size = int(len(image_ids) * rate)
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:]
    train_set = get_coco_by_imgids(coco, train_ids)
    test_set = get_coco_by_imgids(coco, test_ids)
    save_dict(os.path.join(save_dir, 'train.json'), train_set)
    save_dict(os.path.join(save_dir, 'test.json'), test_set)


def main():
    ann_path = '../../work_dirs/data/bottle/annotations/checked_annotations.json'
    save_dir = '../../work_dirs/data/bottle/annotations'
    split_coco(ann_path, save_dir, rate=0.8)

    ann_path = '../../work_dirs/data/fabric/annotations/checked_annotations.json'
    save_dir = '../../work_dirs/data/fabric/annotations'
    split_coco(ann_path, save_dir, rate=0.8)


if __name__ == '__main__':
    main()

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


def split_coco(ann_path, save_dir, rate=0.8, prefix='instance_', random_state=666):
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    random.seed(random_state)
    random.shuffle(image_ids)

    train_size = int(len(image_ids) * rate)
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:]
    train_set = get_coco_by_imgids(coco, train_ids)
    test_set = get_coco_by_imgids(coco, test_ids)
    save_dict(os.path.join(save_dir, '{}train.json'.format(prefix)), train_set)
    save_dict(os.path.join(save_dir, '{}test.json'.format(prefix)), test_set)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Check ann_file')
    parser.add_argument('--ann_file',
                        default='/home/lifeng/undone-work/DefectNet/tools/data/tile/annotations/cut_images_all-check.json',
                        help='annotation file or test image directory')
    parser.add_argument('--save_dir',
                        default='/home/lifeng/undone-work/dataset/detection/tile/annotations/',
                        help='save_dir')
    parser.add_argument('--rate', type=float, default=0.9, help='split rate')
    parser.add_argument('--prefix', default='cut_images_', help='save prefix')
    parser.add_argument('--random_state', type=int, default=666, help='random_state')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    split_coco(args.ann_file, args.save_dir, args.rate, args.prefix, args.random_state)


if __name__ == '__main__':
    main()

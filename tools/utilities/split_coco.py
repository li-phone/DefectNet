import os
import json
import random
from pycocotools.coco import COCO
import copy


def split_coco(ann_path, save_dir, mode='34', rate=0.8, random_state=666):
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    random.seed(random_state)
    random.shuffle(image_ids)

    train_size = int(len(image_ids) * rate)
    train_img_ids = image_ids[:train_size]
    train_image_info = coco.loadImgs(train_img_ids)
    instance_train = copy.deepcopy(coco.dataset)
    instance_train['images'] = train_image_info
    train_img_ids = set(train_img_ids)
    instance_train['annotations'] = [ann for ann in instance_train['annotations'] if ann['image_id'] in train_img_ids]
    save_name = os.path.join(save_dir, 'instance_train_{}.json'.format(mode))
    with open(save_name, 'w') as fp:
        json.dump(instance_train, fp, indent=1, separators=(',', ': '))

    # get no bg annotations for train
    instance_train['categories'] = [ann for ann in instance_train['categories'] if ann['id'] != 0]
    instance_train['annotations'] = [ann for ann in instance_train['annotations'] if ann['category_id'] != 0]
    img_ids = set([ann['image_id'] for ann in instance_train['annotations']])
    instance_train['images'] = [ann for ann in instance_train['images'] if ann['id'] in img_ids]
    assert len(img_ids) == len(instance_train['images'])
    save_name = os.path.join(save_dir, 'instance_train_{}_nobg.json'.format(mode))
    with open(save_name, 'w') as fp:
        json.dump(instance_train, fp, indent=1, separators=(',', ': '))

    test_img_ids = image_ids[train_size:]
    test_image_info = coco.loadImgs(test_img_ids)
    instance_test = copy.deepcopy(coco.dataset)
    instance_test['images'] = test_image_info
    test_img_ids = set(test_img_ids)
    instance_test['annotations'] = [ann for ann in instance_test['annotations'] if ann['image_id'] in test_img_ids]
    save_name = os.path.join(save_dir, 'instance_test_{}.json'.format(mode))
    with open(save_name, 'w') as fp:
        json.dump(instance_test, fp, indent=1, separators=(',', ': '))

    # get no bg annotations for test
    instance_test['categories'] = [ann for ann in instance_test['categories'] if ann['id'] != 0]
    instance_test['annotations'] = [ann for ann in instance_test['annotations'] if ann['category_id'] != 0]
    img_ids = set([ann['image_id'] for ann in instance_test['annotations']])
    instance_test['images'] = [ann for ann in instance_test['images'] if ann['id'] in img_ids]
    assert len(img_ids) == len(instance_test['images'])
    save_name = os.path.join(save_dir, 'instance_test_{}_nobg.json'.format(mode))
    with open(save_name, 'w') as fp:
        json.dump(instance_test, fp, indent=1, separators=(',', ': '))


def main():
    ann_path = '/home/liphone/undone-work/data/detection/fabric/annotations/instance_train,type=34,.json'
    save_dir = '/home/liphone/undone-work/data/detection/fabric/annotations'
    split_coco(ann_path, save_dir, mode='rate={:.2f}'.format(0.8), rate=0.8)


if __name__ == '__main__':
    main()

import os
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize


def load_dict(fname):
    with open(fname, "r") as fp:
        o = json.load(fp, )
        return o


def save_dict(fname, d, mode='w', **kwargs):
    # 持久化写入
    with open(fname, mode, encoding='utf-8') as fp:
        # json.dump(d, fp, cls=NpEncoder, indent=1, separators=(',', ': '))
        json.dump(d, fp, **kwargs)


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def check_coco(src, dst, img_dir=None, replace=True):
    if not replace:
        print('There is an existed {}.'.format(dst))
        return
    coco = load_dict(src)
    cats = json_normalize(coco['categories'])
    cats = cats.sort_values(by='id')
    coco['categories'] = cats.to_dict('records')

    imgs = json_normalize(coco['images'])
    if 'image_id' in list(imgs.columns):
        imgs = imgs.rename(columns={'image_id': 'id'})
    imgs['file_name'] = imgs['file_name'].apply(lambda x: os.path.basename(x))
    imgs = imgs.sort_values(by='id')
    coco['images'] = imgs.to_dict('records')

    if 'annotations' in coco:
        anns = json_normalize(coco['annotations'])
    else:
        ann_fakes = [
            {"area": 100, "iscrowd": 0, "image_id": image['id'], "bbox": [0, 0, 10, 10], "category_id": 1, "id": 1}
            for image in coco['images']
        ]
        anns = json_normalize(ann_fakes)
    anns['id'] = list(range(anns.shape[0]))
    anns = anns.to_dict('records')
    for v in anns:
        if 'segmentation' not in v:
            seg = get_segmentation(v['bbox'])
            v['segmentation'] = [[float(_) for _ in seg]]
    coco['annotations'] = anns
    # check image shape
    if img_dir is not None:
        for i, v in tqdm(enumerate(coco['images'])):
            if os.path.exists(os.path.join(img_dir, v['file_name'])):
                img_ = cv.imread(os.path.join(img_dir, v['file_name']))
                height_, width_, _ = img_.shape
            else:
                row = coco['images'][i]
                height_, width_, _ = int(row['height']), int(row['width']), 3
            assert height_ is not None and width_ is not None
            v['width'] = width_
            v['height'] = height_
    save_dict(dst, coco)
    print('check_coco done!')
    return dst


def check_box(coco, save_name, img_dir):
    if isinstance(coco, str):
        coco = load_dict(coco)
    images = {v['id']: v for v in coco['images']}
    cat2label = {v['id']: v['name'] for v in coco['categories']}
    annotations = {v['id']: v for v in coco['annotations']}
    error_boxes = []
    for k, v in annotations.items():
        b = v['bbox']
        image = images[v['image_id']]
        assert image is not None and image['width'] is not None
        if not (0 <= b[0] <= image['width'] and 0 <= b[1] <= image['height'] and b[2] > 0 and b[3] > 0 \
                and 0 <= b[0] + b[2] <= image['width'] and 0 <= b[1] + b[3] <= image['height']):
            error_boxes.append(v['id'])
    from draw_box import DrawBox
    draw = DrawBox(len(cat2label))

    def save_coco():
        coco['annotations'] = [v for k, v in annotations.items()]
        save_dict(save_name, coco)
        print('save done!')

    def help():
        print('Q: quit, Z: save, X: delete, *: stride\n' \
              'W: up, A: left, S: down, D: right\n' \
              'L: box left, R: box right, T: box top, B: box bottom\n')

    stride = 10
    while len(error_boxes) > 0:
        print('error boxes size: ', len(error_boxes))
        v = annotations[error_boxes[0]]
        b = v['bbox']
        b = [b[0], b[1], b[2] + b[0], b[3] + b[1]]
        image = images[v['image_id']]
        src_img = cv.imread(os.path.join(img_dir, image['file_name']))
        cv.namedWindow('Error_Box', cv.WINDOW_NORMAL)
        direction = 0
        while True:
            img = draw.draw_box(src_img, [b], [cat2label[v['category_id']]])
            show_img = np.array(img).copy()
            cv.imshow("Error_Box", show_img)
            key = cv.waitKeyEx(0)
            if key == 104:
                help()
                break
            elif key == 56:
                try:
                    s = float(input('please input number: '))
                    stride = s
                    print('stride', stride)
                except:
                    print('please input number!')
            elif key == 113:
                error_boxes.pop(0)
                break
            elif key == 120:
                ann_id = error_boxes[0]
                annotations.pop(ann_id)
                error_boxes.pop(0)
                b = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                v['bbox'] = b
                save_coco()
                break
            elif key == 122:
                error_boxes.pop(0)
                b = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                v['bbox'] = b
                save_coco()
                break
            elif key == 108:
                direction = 0
            elif key == 116:
                direction = 1
            elif key == 114:
                direction = 2
            elif key == 98:
                direction = 3
            elif key == 97:
                b[direction] -= stride
                b[direction] = max(b[direction], 0)
            elif key == 119:
                b[direction] -= stride
                b[direction] = max(b[direction], 0)
            elif key == 100:
                b[direction] += stride
                b[direction] = min(b[direction], show_img.shape[1])
            elif key == 115:
                b[direction] += stride
                b[direction] = min(b[direction], show_img.shape[0])
    save_coco()
    print('check_box done!')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Check ann_file')
    parser.add_argument('--ann_file',
                        default='/home/lifeng/undone-work/DefectNet/tools/data/tile/annotations/instance_all.json',
                        help='annotation file or test image directory')
    parser.add_argument('--save_name',
                        default='/home/lifeng/undone-work/DefectNet/tools/data/tile/annotations/instance_all-check.json',
                        help='save_name')
    parser.add_argument('--img_dir',
                        default='"/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/train_imgs/"',
                        help='img_dir')
    parser.add_argument('--check_type', default='coco,box', help='check_type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    check_type = args.check_type.split(',')
    if 'coco' in check_type:
        args.ann_file = check_coco(args.ann_file, args.save_name, args.img_dir)
    if 'box' in check_type:
        check_box(args.ann_file, args.save_name, args.img_dir)


if __name__ == '__main__':
    main()

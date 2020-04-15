import os
import json
from pandas import json_normalize


def load_dict(fname):
    with open(fname, "r") as fp:
        o = json.load(fp, )
        return o


def save_dict(fname, d, mode='w', **kwargs):
    # 持久化写入
    with open(fname, mode) as fp:
        # json.dump(d, fp, cls=NpEncoder, indent=1, separators=(',', ': '))
        json.dump(d, fp, **kwargs)


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def check_w_h(anns):
    for i in range(len(anns) - 1, -1, -1):
        v = anns[i]
        if 'bbox' in v:
            b = v['bbox']
            if b[2] <= 0 or b[3] <= 0:
                anns.pop(i)
    return anns


def check_coco(src, dst, replace=True):
    if not replace:
        print('There is an existed {}.'.format(dst))
        return
    coco = load_dict(src)
    cats = json_normalize(coco['categories'])
    cats = cats.sort_values(by='id')
    cats = cats.to_dict('id')
    coco['categories'] = list(cats.values())

    imgs = json_normalize(coco['images'])
    if 'image_id' in list(imgs.columns):
        imgs = imgs.rename(columns={'image_id': 'id'})
    imgs['file_name'] = [os.path.basename(p) for p in list(imgs['file_name'])]
    imgs = imgs.sort_values(by='id')
    imgs = imgs.to_dict('id')
    coco['images'] = list(imgs.values())

    if 'annotations' in coco:
        anns = json_normalize(coco['annotations'])
    else:
        ann_fakes = [
            {"area": 100, "iscrowd": 0, "image_id": image['id'], "bbox": [0, 0, 10, 10], "category_id": 1, "id": 1}
            for image in coco['images']
        ]
        anns = json_normalize(ann_fakes)
    anns['id'] = list(range(anns.shape[0]))
    anns = anns.to_dict('id')
    for k, v in anns.items():
        if 'segmentation' not in v:
            seg = get_segmentation(v['bbox'])
            v['segmentation'] = [[float(_) for _ in seg]]
    coco['annotations'] = list(anns.values())
    # filter the error boxes
    coco['annotations'] = check_w_h(coco['annotations'])

    save_dict(dst, coco)
    print('Done!')


def main():
    check_coco(
        '../../work_dirs/data/bottle/annotations/annotations.json',
        '../../work_dirs/data/bottle/annotations/checked_annotations.json',
    )


if __name__ == '__main__':
    main()

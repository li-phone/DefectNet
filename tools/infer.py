from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import json
import argparse
import os
import glob
from tqdm import tqdm
from utilities.draw_util import draw_coco


def save_json(results, submit_filename):
    with open(submit_filename, 'w') as fp:
        json.dump(results, fp)


def infer(model, infer_object, img_dir=None, have_bg=False, mask=False):
    images = None
    if isinstance(infer_object, list):
        images = infer_object
    elif isinstance(infer_object, str):
        if infer_object[-5:] == '.json':
            from batch_util import load_json
            coco = load_json(infer_object)
            images = coco['images']
        else:
            img_dir = infer_object
            images = glob.glob(infer_object)
            images = [{'file_name': os.path.basename(p) for p in images}]
    assert images is not None

    results = dict(images=[], annotations=[])
    for i, image in tqdm(enumerate(images)):
        if 'id' in image:
            img_id = image['id']
        elif 'image_id' in image:
            img_id = image['image_id']
        else:
            img_id = i

        results['images'].append(dict(file_name=os.path.basename(image['file_name']), id=img_id))
        img_path = os.path.join(img_dir, os.path.basename(image['file_name']))
        result = inference_detector(model, img_path)
        if mask:
            result = result[0]
        for idx, pred in enumerate(result):
            if have_bg:
                category_id = idx
            else:
                category_id = idx + 1
            if 0 == category_id:
                continue

            for x in pred:
                bbox_pred = {
                    "image_id": img_id,
                    "bbox": [float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1])],
                    "category_id": category_id,
                    "score": float(x[4]),
                }
                results['annotations'].append(bbox_pred)
        # break
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--config',
        default='../config_alcohol/cascade_rcnn_r50_fpn_1x/dig_augment_n4_id3.py',
        help='train config file path')
    parser.add_argument(
        '--resume_from',
        default='../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/dig_augment_n4_id3/epoch_12.pth',
        help='train config file path')
    parser.add_argument(
        '--infer_object',
        default='/home/liphone/undone-work/data/detection/alcohol/test')
    parser.add_argument(
        '--img_dir',
        default='/home/liphone/undone-work/data/detection/alcohol/test')
    parser.add_argument(
        '--work_dir',
        default='../work_dirs/alcohol/cascade_rcnn_r50_fpn_1x/dig_augment_n4_id3/',
        help='train config file path')
    parser.add_argument('--have_bg', default=False)
    parser.add_argument('--mask', default=False)
    args = parser.parse_args()

    return args


def draw(img_dir, work_dir, ann_file, gt_file):
    from utilities.utils import save_dict, load_dict
    coco = load_dict(gt_file)
    label_list = [r['name'] for r in coco['categories']]
    label_list.insert(0, '背景')
    draw_coco(
        ann_file,
        img_dir,
        os.path.join(work_dir, '.infer_tmp'),
        label_list,
    )


def main(**kwargs):
    args = parse_args()
    for k, v in kwargs.items():
        args.__setattr__(k, v)

    model = init_detector(args.config, args.resume_from, device='cuda:0')

    results = infer(model, args.infer_object, args.img_dir, args.have_bg, args.mask)
    save_json(results['annotations'], args.submit_out[:-5] + '.submit.json')
    save_json(results, args.submit_out[:-5] + '.bbox.json')
    from coco2csv import coco2csvsubmit
    coco2csvsubmit(
        gt_result=args.config.data['train']['ann_file'],
        csv_name=args.submit_out[:-5] + '.submit.csv',
        dt_result=args.submit_out[:-5] + '.bbox.json')
    draw(args.img_dir, args.work_dir, args.submit_out[:-5] + '.bbox.json', args.infer_object)


if __name__ == '__main__':
    main()
    print('infer all test images ok!')

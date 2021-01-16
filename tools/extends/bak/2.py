import json
import os
import argparse
import numpy as np

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


def submit_jsons(save_dir, dt_result, keyword=None, image_id='image_id', bbox='bbox', category_id='category_id',
                 **kwargs):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_ids = dt_result[image_id].unique()
    for img_id in img_ids:
        keep_df = dt_result[dt_result[image_id] == img_id]
        if keyword is None:
            d = keep_df.to_dict('records')
            d = [{r[category_id]: r[bbox]} for r in d]
        else:
            d = keep_df.to_dict('records')
        save_name = os.path.join(save_dir, img_id)
        with open(save_name, 'w', encoding='utf-8')as fp:
            json.dump(d, fp)


def convert_type(gt_result, dt_result, filter_id=None, cvt_img_id=None, cvt_box=None, split_box=None, cvt_score=None,
                 cvt_cat_id=None):
    if isinstance(gt_result, str):
        from pycocotools.coco import COCO
        gt_result = COCO(gt_result)
    if isinstance(dt_result, str):
        with open(dt_result) as fp:
            dt_result = json.load(fp)
    if isinstance(dt_result, list):
        dt_result = json_normalize(dt_result)
    else:
        raise Exception('dt_result should be list type !')
    if filter_id is not None:
        dt_result = dt_result[dt_result['category_id'] != filter_id]
    if cvt_img_id is not None:
        img_ids = gt_result.getImgIds()
        imgs = gt_result.loadImgs(img_ids)
        if cvt_img_id == '.':
            m = {img['id']: img['file_name'] for img in imgs}
            dt_result['image_id'] = dt_result['image_id'].apply(lambda x: m[x])
        else:
            m = {img['id']: img['file_name'][:img['file_name'].rfind('.') + 1] + cvt_img_id for img in imgs}
            dt_result['image_id'] = dt_result['image_id'].apply(lambda x: m[x])
    if cvt_box is not None:
        if cvt_box == 'xywh2xyxy':
            boxes = dt_result.pop('bbox')
            boxes = np.array(list(boxes))
            boxes[:, 2:] += boxes[:, :2]
            boxes = [list(map(float, b)) for b in boxes]
            dt_result['bbox'] = boxes
        elif cvt_box == 'xyxy2xywh':
            boxes = dt_result.pop('bbox')
            boxes = np.array(list(boxes))
            boxes[:, 2:] -= boxes[:, :2]
            dt_result['bbox'] = list(boxes)
            boxes = [list(map(float, b)) for b in boxes]
            dt_result['bbox'] = boxes
    if split_box is not None:
        boxes = dt_result.pop('bbox')
        boxes = np.array(list(boxes))
        b0, b1, b2, b3 = list(map(float, boxes[:, 0])), list(map(float, boxes[:, 1])), list(
            map(float, boxes[:, 2])), list(map(float, boxes[:, 3]))
        if cvt_box == 'xywh2xyxy':
            dt_result['xmin'] = b0
            dt_result['ymin'] = b1
            dt_result['xmax'] = b2
            dt_result['ymax'] = b3
        elif cvt_box == 'xyxy2xywh':
            dt_result['x'] = b0
            dt_result['y'] = b1
            dt_result['w'] = b2
            dt_result['h'] = b3
    if cvt_score is not None:
        assert split_box is None
        if cvt_score == 'append':
            score = dt_result.pop('score')
            boxes = dt_result.pop('bbox')
            score = np.array([list(score)])
            boxes = np.array(list(boxes))
            boxes = np.concatenate((boxes, score.T), axis=1)
            boxes = [list(map(float, b)) for b in boxes]
            dt_result['bbox'] = list(boxes)
    if cvt_cat_id is not None:
        cat_ids = gt_result.getCatIds()
        cats = gt_result.loadCats(cat_ids)
        m = {cat['id']: cat['name'] for cat in cats}
        dt_result['category_id'] = dt_result['category_id'].apply(lambda x: m[x])
    return dt_result


def parse_args():
    parser = argparse.ArgumentParser(description='Transform coco submit to other submit format')
    parser.add_argument('--gt_file',
                        default="/home/lifeng/undone-work/dataset/detection/tile/annotations/submit_testA.json",
                        help='annotated file')
    parser.add_argument('--dt_file',
                        default="/data/liphone/defectnet/experiments/work_dirs/tile/baseline_model_cut_ROI_cut_1000x1000/data_mode=test+.bbox.json",
                        help='detected file for list type')
    parser.add_argument('--save_name',
                        default="/data/liphone/defectnet/experiments/work_dirs/tile/baseline_model_cut_ROI_cut_1000x1000/do_submit_testA.json",
                        help='save file or save directory')
    parser.add_argument(
        '--columns',
        nargs='+', action=MultipleKVAction, help='rename dt_file columns')
    parser.add_argument(
        '--convert',
        nargs='+', action=MultipleKVAction,
        help='convert columns format, filter_id=[0], cvt_img_id=[None, ., .xxx], cvt_box=[None, xywh2xyxy, xyxy2xywh], split_box=[None], cvt_score=[None, append], cvt_cat_id=[None]')
    parser.add_argument(
        '--options',
        nargs='+', action=MultipleKVAction, help='jsons fmt: keyword=[None]')
    parser.add_argument(
        '--fmt',
        choices=['None', 'jsons', 'csv'],
        default='None', help='format type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    convert = {'cvt_img_id': '.'} if args.convert is None else args.convert
    options = {} if args.options is None else args.options
    columns = {'image_id': 'name', 'category_id': 'category'} if args.columns is None else args.columns
    dt_result = convert_type(args.gt_file, args.dt_file, **convert)
    for k, v in columns.items():
        dt_result.rename(columns={k: v}, inplace=True)
    if args.fmt == 'csv':
        dt_result.to_csv(args.save_name, index=False)
    elif args.fmt == 'jsons':
        for k, v in columns.items():
            options[k] = v
        submit_jsons(args.save_name, dt_result, **options)
    else:
        d = dt_result.to_dict('records')
        with open(args.save_name, 'w', encoding='utf-8') as fp:
            json.dump(d, fp)


if __name__ == '__main__':
    main()

from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
from PIL import ImageFile
import numpy as np
import os
import pandas as pd
import json
from tqdm import tqdm
import argparse

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True


def imsave(image, image_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(image_path)


def check_contain_chinese(check_str):
    for c in check_str:
        if '\u4e00' <= c <= '\u9fa5':
            return True
    return False


def check_unicode_len(check_str):
    ch_len = 0
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fa5':
            ch_len += 2
        else:
            ch_len += 1
    return ch_len


def pil_rect(image, b, c=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    draw.line(
        [(b[0], b[1]), (b[0], b[3]), (b[2], b[3]), (b[2], b[1]), (b[0], b[1])],
        width=3, fill=c)
    return image


def xyxy2xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def xywh2xyxy(b):
    return [b[0], b[1], b[2] + b[0], b[3] + b[1]]


def random_colors(types_len):
    group_num = types_len ** (1 / 3.)
    group_num = int(group_num + 1)
    group_num = (types_len + group_num) ** (1 / 3.)
    group_num = int(group_num + 1)
    if group_num < 3:
        group_num = 3
    colors = []
    step = int(255 / (group_num - 1))
    rgb_vals = [x for x in range(0, 256, step)]
    for red in rgb_vals:
        for green in rgb_vals:
            for blue in rgb_vals:
                if red == green and red == blue:
                    continue
                colors.append((red, green, blue))
    np.random.shuffle(colors)
    return colors


class DrawBox(object):
    def __init__(self, color_num, box_mode='xyxy'):
        self.color_num = color_num
        self.box_mode = box_mode
        self.color_ind = {}
        self.colors = tuple(random_colors(self.color_num))

        # self.font = ImageFont.truetype('simsun.ttc', self.fontsize, encoding="uti-8")
        # abs_file = __file__.replace('\\', '/')
        # end_idx = abs_file.rfind('/')
        # abs_file = abs_file[:end_idx]
        # font = ImageFont.truetype(os.path.join(abs_file, 'simsun.ttc'), fontsize, encoding='unic')

    def get_color_ind(self, label):
        if label in self.color_ind:
            return self.color_ind[label]
        else:
            while True:
                ind = np.random.randint(len(self.colors))
                if ind not in self.color_ind.values():
                    self.color_ind[label] = ind
                    return self.color_ind[label]

    def draw_box(self, image, boxes, labels, scores=None, fontsize=None, line_width=None):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if fontsize is None:
            fontsize = int(min(image.size) / 1000 * 24)
        if line_width is None:
            line_width = int(min(image.size) / 1000 * 3)
        font = ImageFont.truetype('simsun.ttc', fontsize, encoding="uti-8")

        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        for i, box, lab in zip(range(len(boxes)), boxes, labels):
            box = xywh2xyxy(box) if self.box_mode == 'xywh' else box
            l, t, r, b = box

            # draw bbox
            ind = self.get_color_ind(lab)
            draw.rectangle(xy=((l, t), (r, b)), outline=self.colors[ind], width=line_width)

            # draw label
            if scores is not None:
                lab = '{}:{:.2f}'.format(lab, scores[i])

            character_size = int((1 + check_unicode_len(lab)) * fontsize / 2)
            text_height = 32.0 / 24 * fontsize
            if t - text_height > 0 and l + character_size < im_width:
                pos_x = l
                pos_y = t - text_height
            elif t + text_height < im_height and r + character_size < im_width:
                pos_x = r
                pos_y = t
            elif b + text_height < im_height and l + character_size < im_width:
                pos_x = l
                pos_y = b
            elif t + text_height < im_height and l - character_size > 0:
                pos_x = l - character_size
                pos_y = t
            else:
                pos_x = l
                pos_y = t

            pos_right = pos_x + character_size
            pos_bottom = pos_y + text_height

            bgc = (255 - self.colors[ind][0], 255 - self.colors[ind][1], 255 - self.colors[ind][2])
            draw.rectangle(xy=((pos_x, pos_y), (pos_right, pos_bottom)), fill=bgc)
            txt_color = (0, 0, 0)
            draw.text((pos_x + 4, pos_y + 4), lab, txt_color, font=font)

        return image


def draw_bbox(image, boxes, category_ids, label_list, colors, scores=None, box_mode='xyxy', fontsize=24):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    abs_file = __file__
    abs_file = abs_file.replace('\\', '/')
    end_idx = abs_file.rfind('/')
    abs_file = abs_file[:end_idx]
    # font = ImageFont.truetype(os.path.join(abs_file, 'simsun.ttc'), fontsize, encoding='unic')
    font = ImageFont.truetype('simsun.ttc', 24, encoding="uti-8")
    im_width, im_height = image.size
    for idx, bbox, cat_id in zip(range(len(boxes)), boxes, category_ids):
        if box_mode == 'xyxy':
            (left, top, right, bottom) = (bbox[0], bbox[1], bbox[2], bbox[3])
        elif box_mode == 'xywh':
            (left, top, right, bottom) = (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1])
        else:
            (left, top, right, bottom) = (bbox[0], bbox[1], bbox[2], bbox[3])

        # draw bbox
        cat_id = int(cat_id)
        color = colors[int(cat_id)]
        draw.rectangle(xy=((left, top), (right, bottom)), outline=color, width=3)

        # draw label
        if scores is None:
            label = '{}'.format(label_list[cat_id])
        else:
            label = '{}:{:.2f}'.format(label_list[cat_id], scores[idx])

        character_size = int((1 + check_unicode_len(label)) * fontsize / 2)
        text_height = 32.0 / 24 * fontsize
        if top - text_height > 0 and left + character_size < im_width:
            pos_x = left
            pos_y = top - text_height
        elif top + text_height < im_height and right + character_size < im_width:
            pos_x = right
            pos_y = top
        elif bottom + text_height < im_height and left + character_size < im_width:
            pos_x = left
            pos_y = bottom
        elif top + text_height < im_height and left - character_size > 0:
            pos_x = left - character_size
            pos_y = top
        else:
            pos_x = left
            pos_y = top

        pos_right = pos_x + character_size
        pos_bottom = pos_y + text_height

        bgc = (255 - colors[cat_id][0], 255 - colors[cat_id][1], 255 - colors[cat_id][2])
        draw.rectangle(xy=((pos_x, pos_y), (pos_right, pos_bottom)), fill=bgc)
        bgc = (0, 0, 0)
        draw.text((pos_x + 4, pos_y + 4), label, bgc, font=font)

    return image


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


def parse_args():
    parser = argparse.ArgumentParser(description='Transform other dataset format into coco format')
    parser.add_argument('--ann_file', help='ann_file')
    parser.add_argument('--img_dir', help='img_dir')
    parser.add_argument('--save_dir', help='save_dir')
    parser.add_argument(
        '--options',
        nargs='+', action=MultipleKVAction, help='custom options')
    args = parser.parse_args()
    return args


def draw_coco(anns, img_dir, save_dir, cat2label=None, imgid2filename=None, score_thr=0.05, fontsize=84):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(anns, str):
        with open(anns, "r") as fp:
            anns = json.load(fp)
    if isinstance(anns, dict):
        _cat2label = {v['id']: v['name'] for v in anns['categories']}
        _imgid2filename = {v['id']: v['file_name'] for v in anns['images']}
        anns = json_normalize(anns['annotations'])
        anns['category_id'] = anns['category_id'].apply(lambda x: _cat2label[x])
        anns['image_id'] = anns['image_id'].apply(lambda x: _imgid2filename[x])
    if isinstance(anns, list):
        anns = json_normalize(anns)
    assert isinstance(anns, pd.DataFrame)
    if 'score' in list(anns.columns):
        anns = anns[anns['score'] >= score_thr]

    color_num = len(cat2label) if cat2label is not None else len(anns['category_id'])
    drawbox = DrawBox(color_num=color_num, box_mode='xywh')
    image_ids = anns['image_id'].unique()
    for image_id in tqdm(image_ids):
        file_name = image_id if imgid2filename is None else imgid2filename[image_id]
        keep_ann = anns[anns['image_id'] == image_id]
        if 'score' in list(keep_ann.columns):
            keep_ann.sort_values(by='score', inplace=True)
            scores = list(keep_ann['score'])
        else:
            scores = None
        image = Image.open(os.path.join(img_dir, file_name)).convert('RGB')
        category_id = list(keep_ann['category_id'])
        category_id = [cat2label[v] for v in category_id] if cat2label is not None else category_id
        img2 = drawbox.draw_box(image, list(keep_ann['bbox']), category_id, scores, fontsize)
        imsave(img2, os.path.join(save_dir, file_name + '_draw.jpg'))


def main():
    args = parse_args()
    args.ann_file = '/home/lifeng/undone-work/DefectNet/tools/data/fabric/annotations/instance_all.json'
    args.img_dir = '/home/lifeng/undone-work/DefectNet/tools/data/fabric/trainval'
    args.save_dir = '/home/lifeng/undone-work/DefectNet/tools/data/fabric/coco_draw_1'
    # cn2eng = {
    #     '背景': 'background', '破洞': 'hole', '水渍': 'water stain', '油渍': 'oil stain',
    #     '污渍': 'soiled', '三丝': 'three silk', '结头': 'knots', '花板跳': 'card skip', '百脚': 'mispick',
    #     '毛粒': 'card neps', '粗经': 'coarse end', '松经': 'loose warp', '断经': 'cracked ends',
    #     '吊经': 'buttonhold selvage', '粗维': 'coarse picks', '纬缩': 'looped weft', '浆斑': 'hard size',
    #     '整经结': 'warping knot', '星跳': 'stitch', '跳花': 'skips',
    #     '断氨纶': 'broken spandex', '稀密档': 'thin thick place', '浪纹档': 'buckling place', '色差档': 'color shading',
    #     '磨痕': 'smash', '轧痕': 'roll marks', '修痕': 'take marks', '烧毛痕': 'singeing', '死皱': 'crinked',
    #     '云织': 'uneven weaving', '双纬': 'double pick', '双经': 'double end', '跳纱': 'felter', '筘路': 'reediness',
    #     '纬纱不良': 'bad weft yarn',
    # }
    # cn2eng = {i: v for i, v in enumerate(cn2eng.values())}
    # with open(
    #         '/home/liphone/undone-work/DefectNet/work_dirs/data/fabric/other_important_files/annotations/instance_test_rate=0.80.json') as fp:
    #     test_coco = json.load(fp)
    #     _imgid2filename = {v['id']: v['file_name'] for v in test_coco['images']}

    # kwargs = {'cat2label': cn2eng, 'imgid2filename': _imgid2filename} if args.options is None else args.options
    kwargs = {} if args.options is None else args.options
    draw_coco(args.ann_file, args.img_dir, args.save_dir, **kwargs)


if __name__ == '__main__':
    main()

from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
from PIL import ImageFile
import numpy as np
import os
from pandas import json_normalize
import pandas as pd
import json
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_img(image_path):
    im = Image.open(image_path)
    if im.mode == '1' or im.mode == 'L' or im.mode == 'I' \
            or im.mode == 'F' or im.mode == 'P' or im.mode == 'RGBA' \
            or im.mode == 'CMYK' or im.mode == 'YCbCr':
        im = im.convert('RGB')
    else:
        im = im.convert('RGB')

    return im


def save_img(image, image_path):
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


def draw_bbox(image, boxes, category_ids, label_list, colors, scores=None, box_mode='xyxy', fontsize=24):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    abs_file = __file__
    abs_file = abs_file.replace('\\', '/')
    end_idx = abs_file.rfind('/')
    abs_file = abs_file[:end_idx]
    font = ImageFont.truetype(os.path.join(abs_file, 'simsun.ttc'), fontsize, encoding='unic')
    # font = ImageFont.truetype('simsun.ttc', 24, encoding="uti-8")
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


def set_colors(types_len):
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


def draw_coco(ann_file, img_dir, save_dir, label_list=None, on='image_id', thresh=0., fontsize=16):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(ann_file, str):
        with open(ann_file, "r") as fp:
            anns = json.load(fp, )
    else:
        anns = ann_file
    if label_list is None:
        if 'categories' in anns:
            label_list = [None] * (len(anns['categories']) + 1)
            for i, v in enumerate(anns['categories']):
                label_list[v['id']] = v['name']

    images = json_normalize(anns['images'])
    if 'id' in list(images.columns):
        images.rename(columns={'id': 'image_id'}, inplace=True)
    annotations = json_normalize(anns['annotations'])
    results = pd.merge(annotations, images, on=on)
    columns = list(results.columns)
    if 'score' in columns:
        results = results[results['score'] >= thresh]
    colors = set_colors(len(label_list))
    file_names = results['file_name'].unique()
    for file_name in tqdm(file_names):
        result = results[results['file_name'] == file_name]
        image = read_img(os.path.join(img_dir, file_name))
        if 'score' not in columns:
            scores = None
        else:
            scores = list(result['score'])
        img_pred = draw_bbox(
            image, list(result['bbox']), list(result['category_id']), label_list, colors,
            scores=scores, box_mode='xywh', fontsize=fontsize, )
        save_img(img_pred, os.path.join(save_dir, file_name + '_draw.jpg'))

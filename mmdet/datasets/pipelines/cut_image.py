import cv2
import numpy as np
import copy

from ..registry import PIPELINES


@PIPELINES.register_module
class CutImage(object):

    def __init__(self, window=(2666, 1600), step=(1333, 800), order_index=True, is_keep_none=False):
        self.window = window
        self.step = step
        self.order_index = {} if order_index else None
        self.is_keep_none = is_keep_none

    def __call__(self, results):
        img_h, img_w, _ = results['img'].shape
        cut_cnt = (img_h // self.step[1] + 1) * (img_w // self.step[0] + 1)
        cut_results = []
        for i in range(0, img_h, self.step[1]):
            for j in range(0, img_w, self.step[0]):
                h = min(img_h, i + self.window[1])
                w = min(img_w, j + self.window[0])
                img = results['img'][i:h, j:w, :]
                result = copy.deepcopy(results)
                for k, v in result.items():
                    result[k] = copy.deepcopy(results[k])
                result['top_left'] = (j, i)
                result['img_info']['height'] = img.shape[0]
                result['img_info']['width'] = img.shape[1]
                result['ann_info']['bboxes'][:, 0] -= result['top_left'][0]
                result['ann_info']['bboxes'][:, 2] -= result['top_left'][0]
                result['ann_info']['bboxes'][:, 1] -= result['top_left'][1]
                result['ann_info']['bboxes'][:, 3] -= result['top_left'][1]
                result['img'] = img
                result['no_cut_img_shape'] = (img_h, img_w, _)
                result['img_shape'] = img.shape
                result['ori_shape'] = img.shape
                result['gt_bboxes'][:, 0] -= result['top_left'][0]
                result['gt_bboxes'][:, 2] -= result['top_left'][0]
                result['gt_bboxes'][:, 1] -= result['top_left'][1]
                result['gt_bboxes'][:, 3] -= result['top_left'][1]
                h_, w_, _ = img.shape
                bboxes = result['ann_info']['bboxes']
                keep_idx = [i for i in range(len(bboxes)) if bboxes[i][0] >= 0
                            and bboxes[i][1] >= 0 and bboxes[i][2] < w_ and bboxes[i][3] < h_]
                result['ann_info']['bboxes'] = result['ann_info']['bboxes'][keep_idx]
                result['ann_info']['labels'] = result['ann_info']['labels'][keep_idx]
                # result['ann_info']['bboxes_ignore'] = result['ann_info']['bboxes_ignore'][keep_idx]
                result['gt_bboxes'] = result['gt_bboxes'][keep_idx]
                result['gt_labels'] = result['gt_labels'][keep_idx]
                # result['gt_bboxes_ignore'] = result['gt_bboxes_ignore'][keep_idx]
                if self.is_keep_none:
                    cut_results.append(result)
                else:
                    if len(result['gt_bboxes']) > 0 and len(result['gt_labels']) > 0:
                        cut_results.append(result)
        if len(cut_results) < 1:
            return None
        if self.order_index is not None:
            key = results['filename']
            if key not in self.order_index:
                self.order_index[key] = 0
            else:
                self.order_index[key] = (self.order_index[key] + 1) % len(cut_results)
            return cut_results[self.order_index[key]]
        return cut_results

    def __repr__(self):
        return self.__class__.__name__ + '(window={})'.format(
            self.window)


def debug_detect_max_rect():
    sp = CutImage((2666 * 2, 1600 * 2), (2666, 1600))
    image = cv2.imread("C:/Users/97412/Pictures/220_140_t20201124140233485_CAM2.jpg")
    results = sp.__call__(image)


def debug():
    debug_detect_max_rect()


if __name__ == '__main__':
    debug()

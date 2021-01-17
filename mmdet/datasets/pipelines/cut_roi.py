import heapq
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

from ..registry import PIPELINES


@PIPELINES.register_module
class CutROI(object):

    def __init__(self, training=True, padding=50, threshold='ostu', method='findContours'):
        self.training = training
        self.threshold = threshold
        self.method = method
        self.padding = padding

    @staticmethod
    def get_intersection(a, b):
        y = (a[1] * np.cos(b[0]) - b[1] * np.cos(a[0])) / (np.sin(a[0]) * np.cos(b[0]) - np.sin(b[0]) * np.cos(a[0]))
        x = (a[1] * np.sin(b[0]) - b[1] * np.sin(a[0])) / (np.cos(a[0]) * np.sin(b[0]) - np.cos(b[0]) * np.sin(a[0]))
        return (x, y)

    @staticmethod
    def lines2rect(lines, min_angle=2):
        res = {}
        for x1, y1, x2, y2 in lines[:]:
            radian = np.arctan((x1 - x2) / (y2 - y1))
            if np.isnan(radian):
                continue
            dist = x1 * np.cos(radian) + y1 * np.sin(radian)
            th = int((radian * 180 / np.pi) // min_angle)
            if th not in res:
                res[th] = []
            res[th].append([radian, dist])
        res_counter = [[len(v), k] for k, v in res.items()]
        topk = heapq.nlargest(2, res_counter, key=lambda x: x)
        if len(topk) < 2:
            return None, None
        min_k, max_k = topk[0][1], topk[1][1]
        r1, r2 = np.array(res[min_k]), np.array(res[max_k])
        r1_min_idx, r1_max_idx = np.argmin(r1[:, 1]), np.argmax(r1[:, 1])
        r2_min_idx, r2_max_idx = np.argmin(r2[:, 1]), np.argmax(r2[:, 1])
        l, r, t, b = r1[r1_min_idx], r1[r1_max_idx], r2[r2_min_idx], r2[r2_max_idx]
        if l is None or r is None or t is None or b is None:
            return None, None
        p1 = CutROI.get_intersection(l, t)
        p2 = CutROI.get_intersection(t, r)
        p3 = CutROI.get_intersection(r, b)
        p4 = CutROI.get_intersection(b, l)
        rect = (min(p1[0], p2[0], p3[0], p4[0]), min(p1[1], p2[1], p3[1], p4[1]),
                max(p1[0], p2[0], p3[0], p4[0]), max(p1[1], p2[1], p3[1], p4[1]))
        if sum(np.isnan(rect)) or sum(np.isinf(rect)):
            return None, None
        return rect, (p1, p2, p3, p4)

    @staticmethod
    def cut_max_rect(image, threshold='ostu', method='findContours', **kwargs):
        if isinstance(image, str):
            image = cv2.imread(image)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if isinstance(threshold, str) and threshold == 'ostu':
            thr, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            # from uuid import uuid4
            # cv2.imwrite("tmp/{}_.jpg".format(uuid4()), img)
            # plt.imshow(img)
            # plt.show()
        elif str(threshold).isdigit():
            thr = float(threshold)
        else:
            thr = 50
        if method == 'findContours':
            if threshold != 'ostu':
                print('Warning!!! findContours method must ostu threshold!')
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) < 1:
                return None, None
            max_size, max_area = 0, 0
            size_idx, area_idx = 0, 0
            for i in range(len(contours)):
                if len(contours[size_idx]) < len(contours[i]):
                    size_idx = i
                    max_size = len(contours[i])
                area_ = cv2.contourArea(contours[i])
                if area_ > max_area:
                    area_idx = i
                    max_area = area_
            x, y, w, h = cv2.boundingRect(contours[area_idx])
            rect = [x, y, x + w, y + h]
            # kwargs['area'].append(max_area)
            # cv2.drawContours(image, contours, area_idx, (0, 255, 0), thickness=15)
            # rect = [int(x) for x in rect]
            # cv2.rectangle(image, tuple(rect[:2]), tuple(rect[2:]), (255, 0, 0), thickness=10)
            # plt.imshow(image)
            # plt.show()
            return rect, None
        elif method == 'HoughLinesP':
            canny_img = cv2.Canny(img, thr, 255)
            h, w = canny_img.shape
            minLineLength = int(min(w, h) / 2)
            maxLineGap = int(np.sqrt(w * w + h * h))
            lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, int(minLineLength / 10),
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
            if lines is None or len(lines) < 1:
                return None, None
            lines = lines[:, 0, :]  # 提取为二维
            rect, pts = CutROI.lines2rect(lines)
            return rect, pts
        else:
            raise Exception("No such {} implement method!".format(method))

    def __call__(self, results):
        results['no_roi_img_shape'] = results['img'].shape
        rect, pts = CutROI.cut_max_rect(results['img'], self.threshold, self.method)
        if rect is None:
            return None
        if self.padding:
            ori_h, ori_w, ori_c = results['img'].shape
            rect = [max(0, rect[0] - self.padding), max(0, rect[1] - self.padding),
                    min(ori_w - 1, rect[2] + self.padding), min(ori_h - 1, rect[3] + self.padding), ]

        x1, y1, x2, y2 = [int(x) for x in rect]
        img = results['img'][y1:y2, x1:x2, :]
        results['roi_top_left'] = [rect[0], rect[1]]
        results['img_info']['height'] = img.shape[0]
        results['img_info']['width'] = img.shape[1]
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        if self.training:
            results['ann_info']['bboxes'][:, 0] -= results['roi_top_left'][0]
            results['ann_info']['bboxes'][:, 2] -= results['roi_top_left'][0]
            results['ann_info']['bboxes'][:, 1] -= results['roi_top_left'][1]
            results['ann_info']['bboxes'][:, 3] -= results['roi_top_left'][1]
            results['gt_bboxes'] = results['ann_info']['bboxes']
        # # test draw cut roi
        # from mmdet.third_party.draw_box import DrawBox
        # import os
        # drawBox = DrawBox(color_num=7)
        # image = drawBox.draw_box(results['img'], results['gt_bboxes'], results['gt_labels'])
        # image = np.array(image)
        # cv2.imwrite("tmp/{}".format(os.path.basename(results['filename'])), image)
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(method={})'.format(
            self.method)


if __name__ == '__main__':
    import glob
    from tqdm import tqdm

    images = glob.glob("/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/*.jpg")
    area = []
    for file in tqdm(images):
        image = cv2.imread(file)
        CutROI.cut_max_rect(image, area=area)
    area = np.array(area)
    print('min:{},max:{},avg:{}'.format(np.min(area), np.max(area), np.mean(area)))

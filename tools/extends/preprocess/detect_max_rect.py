import cv2
import heapq
import numpy as np


def get_intersection(a, b):
    y = (a[1] * np.cos(b[0]) - b[1] * np.cos(a[0])) / (np.sin(a[0]) * np.cos(b[0]) - np.sin(b[0]) * np.cos(a[0]))
    x = (a[1] * np.sin(b[0]) - b[1] * np.sin(a[0])) / (np.cos(a[0]) * np.sin(b[0]) - np.cos(b[0]) * np.sin(a[0]))
    return (x, y)


def _max_rect(lines, min_angle=2):
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
    p1 = get_intersection(l, t)
    p2 = get_intersection(t, r)
    p3 = get_intersection(r, b)
    p4 = get_intersection(b, l)
    rect = (min(p1[0], p2[0], p3[0], p4[0]), min(p1[1], p2[1], p3[1], p4[1]),
            max(p1[0], p2[0], p3[0], p4[0]), max(p1[1], p2[1], p3[1], p4[1]))
    if sum(np.isnan(rect)) or sum(np.isinf(rect)):
        return None, None
    return rect, (p1, p2, p3, p4)


def detect_max_rect(image, threshold='ostu', method='HoughLinesP'):
    if isinstance(image, str):
        image = cv2.imread(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if isinstance(threshold, str) and threshold == 'ostu':
        thr, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
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
        max_ind = 0
        for i in range(len(contours)):
            if len(contours[max_ind]) < len(contours[i]):
                max_ind = i
        rect = cv2.boundingRect(contours[max_ind])
        return rect, None
    elif method == 'HoughLinesP':
        canny_img = cv2.Canny(img, thr, 255)
        w, h = canny_img.shape
        minLineLength = int(min(w, h) / 2)
        maxLineGap = int(np.sqrt(w * w + h * h))
        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, int(minLineLength / 10),
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is None or len(lines) < 1:
            return None, None
        lines = lines[:, 0, :]  # 提取为二维
        rect, pts = _max_rect(lines)
        return rect, pts
    else:
        raise Exception("No such {} implement method!".format(method))


def debug_detect_max_rect():
    import glob
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    def _draw(img):
        rect, pts = detect_max_rect(img)
        if rect is not None:
            x1, y1, x2, y2 = [int(x) for x in rect]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if pts is not None:
                for i in range(len(pts)):
                    c = [int(x) for x in pts[i]]
                    cv2.circle(img, (c[0], c[1]), 5, (0, 255, 0), 2)
        return img

    def check_dir(dir1):
        for path in tqdm(glob.glob(dir1)):
            rect, pts = detect_max_rect(path, method='findContours')
            if rect is None:
                print(path, "can not detect rect!")

    # dir1 = "/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/train_imgs/*"
    # dir2 = "/home/lifeng/undone-work/dataset/detection/tile/tile_round1_testA_20201231/testA_imgs/*"
    # check_dir(dir2) and check_dir(dir1)

    image = cv2.imread("C:/Users/97412/Pictures/220_140_t20201124140233485_CAM2.jpg")
    cv2.imwrite("220_140_t20201124140233485_CAM2_detect.jpg", _draw(image))
    cap = cv2.VideoCapture(0)
    cap.set(3, 300)
    cap.set(4, 300)
    while True:
        ret, frame = cap.read()
        cv2.imshow("detect", _draw(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def debug():
    debug_detect_max_rect()


if __name__ == '__main__':
    debug()

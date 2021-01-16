from mmdet.datasets.pipelines.cut_roi import CutROI
from tqdm import tqdm
import cv2
import glob
import matplotlib.pyplot as plt


def show_img(img):
    plt.imshow(img)
    plt.show()


files = glob.glob("/home/lifeng/undone-work/dataset/detection/tile/tile_round1_train_20201231/train_imgs/*.jpg")
for file in tqdm(files):
    image = cv2.imread(file)
    rect, _ = CutROI.cut_max_rect(image)
    ori_h, ori_w, ori_c = image.shape
    padding = 50
    rect = [max(0, rect[0] - padding), max(0, rect[1] - padding),
            min(ori_w - 1, rect[2] + padding), min(ori_h - 1, rect[3] + padding), ]
    x1, y1, x2, y2 = rect
    show_img(image)
    roi_img = image[y1:y2, x1:x2, :]
    show_img(roi_img)
    gray_img = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    show_img(gray_img)
    thr, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    show_img(img)
    canny_img = cv2.Canny(gray_img, thr, 255)
    show_img(canny_img)
    # show_img(gray_img)
    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1:
        continue
    continue
    for i in range(len(contours)):
        roi = cv2.boundingRect(contours[i])
        roi = [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]
        roi_img = img[roi[1]:roi[3], roi[0]:roi[2], :]
        label = ImageClassification(roi_img)
        if label == 0:
            continue

from pycocotools.coco import COCO
import numpy as np
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def get_ious(anns):
    ious = []
    iou_nums = [0] * 4
    for k, ann in anns.items():
        b = ann['bbox']
        m = min(b[2], b[3])
        if m < 40:
            ious.append(0.2)
            iou_nums[0] += 1
        elif m < 120:
            ious.append(m / 200)
            iou_nums[1] += 1
        elif m < 420:
            ious.append(m / 1500 + 0.52)
            iou_nums[2] += 1
        else:
            ious.append(0.8)
            iou_nums[3] += 1
    return ious, iou_nums


def kmeans(x, n=3):
    x = np.array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    # 假如我要构造一个聚类数为3的聚类器
    estimator = KMeans(n_clusters=n)  # 构造聚类器
    estimator.fit(x)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    return centroids


def con(args):
    cons = []
    for i, v in enumerate(args):
        cons.append({'type': 'ineq', 'fun': lambda x: x[i] - v[0]})
        cons.append({'type': 'ineq', 'fun': lambda x: -x[i] + v[1]})
    cons = tuple(cons)
    return cons


# def get_cluster(x, n=3):
#     centroids = kmeans(x, n)
#     # centroids = np.sort(centroids.reshape(-1))
#     print('=' * 24, 'get_cluster', '=' * 24, '\n', centroids)
#     return centroids


# def iou_cluster(anns, n=3):
#     ious, iou_nums = get_ious(anns)
#     get_cluster(ious, 'iou.png', n=n)


def anchor_cluster(anns, n=3):
    if isinstance(anns, str):
        coco = COCO(anns)
        anns = coco.dataset['annotations']
    elif isinstance(anns, dict):
        anns = anns['annotations']

    boxes = [a['bbox'] for a in anns]
    boxes = np.array(boxes)
    aspect_ratio = boxes[:, 3] / boxes[:, 2]
    centroids = kmeans(aspect_ratio, n)
    centroids = centroids.squeeze()
    centroids = np.sort(centroids, axis=0)
    return list(centroids)
    # hor_ver_ratio = boxes[:, 2] / boxes[:, 3]
    # get_cluster(hor_ver_ratio, 'hor_ver_ratio.png', n=n)


def box_cluster(anns, n=3, sind=2, eind=4):
    if isinstance(anns, str):
        coco = COCO(anns)
        anns = coco.dataset['annotations']
    elif isinstance(anns, dict):
        anns = anns['annotations']

    boxes = [a['bbox'] for a in anns]
    boxes = np.array(boxes)
    # import pandas as pd
    # box_df = pd.DataFrame(data=boxes, columns=['x', 'y', 'w', 'h'])
    # box_df.plot(kind="scatter", x="w", y="h", alpha=0.1)
    # plt.show()
    boxes = boxes[:, sind:eind]
    centroids = kmeans(boxes, n, )
    centroids = np.sort(centroids, axis=0)
    return list(centroids)


def main():
    # name2label = {1: 1, 9: 2, 5: 3, 3: 4, 4: 5, 0: 6, 2: 7, 8: 8, 6: 9, 10: 10, 7: 11}
    # label_weight = {0: 0, 1: 0.15, 2: 0.09, 3: 0.09, 4: 0.05, 5: 0.13, 6: 0.05, 7: 0.12, 8: 0.13, 9: 0.07, 10: 0.12}
    # label2name = {v: k for k, v in name2label.items()}
    # label_weight = {label2name[k]:v for k,v in name_weight.items()}

    ann_file = '/home/lifeng/undone-work/DefectNet/tools/data/fabric/annotations/instance_train.json'
    box_cluster(ann_file, n=10)
    pass


if __name__ == '__main__':
    main()

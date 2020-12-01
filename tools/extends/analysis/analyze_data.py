import seaborn as sns
import matplotlib.pyplot as plt
import json
from pandas.io.json import json_normalize
import os
import pandas as pd
import numpy as np

sns.set(style="darkgrid")


def read_json(json_path):
    results = []
    np_names = [
        'AP', 'AP:0.50', 'AP:0.75', 'AP:S', 'AP:M', 'AP:L',
        'AR', 'AR:0.50', 'AR:0.75', 'AR:S', 'AR:M', 'AR:L',
    ]
    with open(json_path) as fp:
        lines = fp.readlines()
        for line in lines:
            r = json.loads(line)
            d = r['data']['bbox']['data']
            result = dict(cfg=r['cfg'], uid=r['uid'], mode=r['mode'])
            for k, v in zip(np_names, d['coco_eval']):
                result[k] = v
            for k, v in d['classwise'].items():
                result[k] = v
            for k1, v1 in d['defect_eval'].items():
                if isinstance(v1, list):
                    result[k1] = np.mean(v1)
                elif isinstance(v1, dict):
                    for k2, v2 in v1['macro avg'].items():
                        result[k2] = v2
            results.append(result)
    return results


def phrase_json(json_path):
    save_path = json_path[:-5] + '.csv'
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    results = read_json(json_path)
    df = json_normalize(results)
    df.to_csv(save_path, index=False)
    return df


def get_sns_data(data, x_name, y_names, type):
    x, y, hue = np.empty(0), np.empty(0), np.empty(0)
    for y_name in y_names:
        x = np.append(x, data[x_name])
        y = np.append(y, data[y_name])
        hue = np.append(hue, [type[y_name]] * data[y_name].shape[0])
    return pd.DataFrame(dict(x=x, y=y, type=hue))


def lineplot(sns_data, new_x, new_y, ax=None, markers=True):
    sns_data = sns_data.rename(columns={'x': new_x, 'y': new_y})
    ax = sns.lineplot(
        ax=ax,
        x=new_x, y=new_y,
        hue="type",
        style="type",
        markers=markers,
        dashes=False,
        data=sns_data,
        ci=None
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    return ax


def draw_figure(json_path, save_path, x_name):
    save_path = save_path[:-4]
    save_path = save_path.replace('\\', '/')
    save_dir = save_path[:save_path.rfind('/')]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = phrase_json(json_path)
    data = data
    sns.set(style="darkgrid")
    # ids = []
    # for i in range(data.shape[0]):
    #     r = data.iloc[i]
    #     arrs = r['cfg'].split('_')
    #     ids.append(float(arrs[1][:-1]))

    data[x_name] = data['uid']
    data = data[data['mode'] == 'test']

    fig = plt.figure(figsize=(6.4 * 3, 4.8))
    axs = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    # draw_ap_weight
    y_names = ['AP', 'AP:0.50']
    type = {'AP': 'IoU=0.50:0.95', 'AP:0.50': 'IoU=0.50'}
    sns_data = get_sns_data(data, x_name, y_names, type)
    new_x, new_y = x_name, 'average precision'
    lineplot(sns_data, new_x, new_y, axs[0])

    # draw_f1_score_weight
    y_names = ['f1-score']
    type = {'f1-score': 'f1-score'}
    sns_data = get_sns_data(data, x_name, y_names, type)
    new_x, new_y = x_name, 'macro average f1-score'
    lineplot(sns_data, new_x, new_y, axs[1])

    # draw_speed_weight
    y_names = ['fps', 'defect_fps', 'normal_fps']
    type = dict(fps='all images', defect_fps='defect images', normal_fps='normal images')
    sns_data = get_sns_data(data, x_name, y_names, type)
    new_x, new_y = x_name, 'average time(ms)'
    lineplot(sns_data, new_x, new_y, axs[2])

    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.subplots_adjust(left=0.05, right=0.97)
    plt.savefig(save_path + '.svg')
    plt.savefig(save_path + '.eps')
    plt.savefig(save_path + '.jpg')
    plt.show()


def count_data(ann_file, head=None):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    defect_nums = np.empty(0, dtype=int)
    for image in coco.dataset['images']:
        cnt = 0
        annIds = coco.getAnnIds(imgIds=image['id'])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['category_id'] != 0:
                cnt += 1
        defect_nums = np.append(defect_nums, cnt)
    normal_shape = np.where(defect_nums == 0)[0]
    if head is not None:
        print(head + ':\n')
    all_cnt, normal_cnt = len(coco.dataset['images']), normal_shape.shape[0]
    defect_cnt = defect_nums.shape[0] - normal_shape.shape[0]
    print('All images count:', all_cnt)
    print('Normal images count:', normal_cnt)
    print('Defect images count:', defect_cnt)
    print('Normal images : Defect images is ', normal_cnt / defect_cnt)


def draw_avg_infer_time_and_efficient():
    t = 0.5
    tau = [t] * 50000
    rs = np.linspace(0, 50, 50000)
    avg_t = [1 - (1 - t) / (1 + 1 / r) for r in rs]
    e = [(1 - t) / (1 + 1 / r) for r in rs]
    data = pd.DataFrame({'r': rs, 't': avg_t, 'e': e, 'τ': tau})

    y_names = ['t', 'e', 'τ']
    type = {'t': 't', 'e': 'e', 'τ': 'τ'}
    x_name = 'r'
    sns_data = get_sns_data(data, x_name, y_names, type)
    new_x, new_y = x_name, ''
    ax = lineplot(sns_data, new_x, new_y, markers=False)
    plt.subplots_adjust(left=0.1, right=0.925)
    plt.savefig('../results/imgs/draw_avg_infer_time_and_efficient.jpg')
    plt.savefig('../results/imgs/draw_avg_infer_time_and_efficient.svg')
    plt.savefig('../results/imgs/draw_avg_infer_time_and_efficient.eps')
    plt.show()


def main():
    cfg_dir = '../config_fabric/cascade_rcnn_r50_fpn_1x'
    rst_dir = '../results/imgs'
    data_root = '/home/liphone/undone-work/data/detection/fabric'

    # one model
    draw_figure(
        json_path=cfg_dir + '/different_threshold_test,background=No,.json',
        save_path=rst_dir + '/different_threshold_test,threshold=0.00-0.99,background=No,.jpg',
        x_name='threshold',
    )

    # two model

    # defect network
    count_data(data_root + '/annotations/instances_train_20191223_annotations.json', 'all')
    count_data(data_root + '/annotations/instance_train_fabric.json', 'train')
    count_data(data_root + '/annotations/instance_test_fabric.json', 'test')

    # draw_figure(
    #     json_path=cfg_dir + '/different_dfn_weight,background=No,.json',
    #     save_path=rst_dir + '/different_dfn_weight,threshold=0.00-2.00,background=No,.jpg',
    #     x_name='defect finding weight'
    # )

    draw_figure(
        cfg_dir + '/different_ratio_test,background=No,.json',
        rst_dir + '/different_ratio_test,ratio=0.00-12.00,background=No,.jpg',
        x_name='normal : defective'
    )

    draw_avg_infer_time_and_efficient()


if __name__ == '__main__':
    main()

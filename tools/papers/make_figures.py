import os
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use('fast')


def save_plt(save_name, file_types=None):
    if file_types is None:
        file_types = ['.svg', '.jpg']
    save_dir = save_name[:save_name.rfind('/')]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for t in file_types:
        plt.savefig(save_name[:-4] + t)


def line_plot(ax, x, ys, labels=None, styles=None, param_dict=None):
    """
    A helper function to make a line graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    x : array
       The x data

    ys : array
       The y data

    labels : list
       The ys labels

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    for i, y in enumerate(ys):
        if labels is not None:
            if styles is not None:
                ax.plot(x, y, label=labels[i], linestyle=styles[i])
            else:
                ax.plot(x, y, label=labels[i])
        else:
            ax.plot(x, y)
    if param_dict is not None:
        ax.set_xlabel(param_dict['xlabel'])  # Add an x-label to the axes.
        ax.set_ylabel(param_dict['ylabel'])  # Add a y-label to the axes.
        ax.set_title(param_dict['title'])  # Add a title to the axes.
    ax.legend()  # Add a legend
    return ax


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
            d = r['data']
            if isinstance(r['uid'], str):
                x_label = float(r['uid'].split('=')[1])
            elif isinstance(r['uid'], float):
                x_label = r['uid']
            elif isinstance(r['uid'], int):
                x_label = r['uid']
            else:
                raise Exception('No such type for {}!'.format(r['uid']))
            result = dict(cfg=r['cfg'], uid=x_label, mode=r['mode'])
            for k, v in zip(np_names, d['coco_result']['coco_eval']):
                result[k] = v
            for k, v in d['coco_result']['classwise'].items():
                result[k] = v
            for k1, v1 in d['defect_result'].items():
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


def make_figure3():
    # make Figure 3
    fig = plt.figure(figsize=(6.4 * 3, 4.8))
    axes = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
    a = np.linspace(0, 1, 101)
    bs = [0.75, 0.50, 0.25]
    titles = ['0.5<b<=1', 'b=0.5', '0<=b<0.5']
    for ax, b, title in zip(axes, bs, titles):
        t = 1 - (1 - b) * a
        e = (1 - b) * a
        y_eq_b = np.linspace(b, b, len(a))
        y_eq_1_minus_b = np.linspace(1 - b, 1 - b, len(a))
        intersect = (1 / (2 * (1 - b)), 1 / 2)
        if b == 0.5:
            x, ys = a, [t, e, y_eq_b]
            labels, ax_text = ['t', 'e', 'b=0.5'], {'xlabel': 'a', 'ylabel': 'y', 'title': title}
            styles = [None, None, '-.']
        else:
            x, ys = a, [t, e, y_eq_b, y_eq_1_minus_b]
            labels, ax_text = ['t', 'e', 'y=b', 'y=1-b'], {'xlabel': 'a', 'ylabel': 'y', 'title': title}
            styles = [None, None, '-.', '-.']
        line_plot(ax, x, ys, labels, styles, ax_text)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle='--')
    plt.subplots_adjust(left=0.05, right=0.97)
    save_plt('./figures/Figure_3.Detection_efficiency_on_a.jpg')
    plt.show()


def make_evaluation_figure(data_path, save_name, ap_param, f1_score_param, att_param):
    data = phrase_json(data_path)
    fig = plt.figure(figsize=(6.4 * 3, 4.8))
    axes = [fig.add_subplot(1, 3, i) for i in range(1, 4)]

    # draw_ap
    x, ys, labels = data['uid'], [data['AP']], ['AP']
    ax = axes[0]
    line_plot(ax, x, ys, labels, param_dict=ap_param)
    ax.grid(linestyle='--')

    # draw_f1_score
    x, ys, labels = data['uid'], [data['f1-score']], ['F1-score']
    ax = axes[1]
    line_plot(ax, x, ys, labels, param_dict=f1_score_param)
    ax.grid(linestyle='--')

    # draw_average_test_time
    x, ys = data['uid'], [data['att'], data['att_normal'], data['att_defective']]
    labels = ['att', 'att_normal', 'att_defective']
    ax = axes[2]
    line_plot(ax, x, ys, labels, param_dict=att_param)
    ax.grid(linestyle='--')

    plt.subplots_adjust(left=0.05, right=0.97)
    save_plt(save_name)
    plt.show()


def main():
    # make_figure3()

    # make figure 4
    ap_param = {'xlabel': 'score_threshold', 'ylabel': 'average precision', 'title': 'Detecting Defects Performance'}
    f1_score_param = {'xlabel': 'score_threshold', 'ylabel': 'F1-score', 'title': 'Finding Defects Performance'}
    att_param = {'xlabel': 'score_threshold', 'ylabel': 'average test time(ms)', 'title': 'Test Speed Performance'}
    make_evaluation_figure(
        '../../work_dirs/bottle/one_model_cascade_rcnn_r50_fpn_1x/one_model_cascade_rcnn_r50_fpn_1x_score_threshold_test.json',
        './figures/Evaluation_on_different_score_thr_one_model.jpg',
        ap_param, f1_score_param, att_param
    )

    # make figure 5
    ap_param = {'xlabel': 'w', 'ylabel': 'average precision', 'title': 'Detecting Defects Performance'}
    f1_score_param = {'xlabel': 'w', 'ylabel': 'F1-score', 'title': 'Finding Defects Performance'}
    att_param = {'xlabel': 'w', 'ylabel': 'average test time(ms)', 'title': 'Test Speed Performance'}
    make_evaluation_figure(
        '../../work_dirs/bottle/defectnet_constant_cascade_rcnn_r50_fpn_1x/const_weight=0.00/defectnet_constant_cascade_rcnn_r50_fpn_1x_find_best_weight_test.json',
        './figures/Evaluation_on_increasing_loss_weight.jpg',
        ap_param, f1_score_param, att_param
    )


if __name__ == "__main__":
    main()

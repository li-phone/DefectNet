import json
import numpy as np

try:
    from pandas import json_normalize
except:
    from pandas.io.json import json_normalize


def read_json(json_path):
    results = []
    with open(json_path) as fp:
        lines = fp.readlines()
        for line in lines:
            r = json.loads(line)
            d = r['data']['defect_result']
            result = dict()
            for k1, v1 in d.items():
                if isinstance(v1, list):
                    result[k1] = v1
            results.append(result)
    return results


def main():
    data_path = '/home/liphone/undone-work/DefectNet/work_dirs/bottle/defectnet_exponent_cascade_rcnn_r50_fpn_1x/defectnet_exponent_cascade_rcnn_r50_fpn_1x_test.json'
    data = read_json(data_path)
    att, att_d, att_n = [], [], []
    for i, v in enumerate(data):
        att.extend(v['att'])
        att_d.extend(v['att_defective'])
        att_n.extend(v['att_normal'])
    print('avg att: {}, avg att_defective: {}, avg att_normal: {}'.format(np.mean(att), np.mean(att_d), np.mean(att_n)))


if __name__ == '__main__':
    main()

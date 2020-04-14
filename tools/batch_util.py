import json


def load_json(f):
    with open(f) as fp:
        return json.load(fp)


def save_json(o, f):
    with open(f, 'w') as fp:
        return json.dump(o, fp)


def filter_boxes(anns, threshold=0.05):
    for i in range(len(anns) - 1, -1, -1):
        ann = anns[i]
        if ann['score'] < threshold:
            anns.pop(i)
    return anns


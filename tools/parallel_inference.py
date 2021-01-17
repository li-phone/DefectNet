from pycocotools.coco import COCO
from tqdm import tqdm
import os
import json
import threading
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets.pipelines import Compose
import numpy as np
import matplotlib.pyplot as plt
import cv2


class CutConfig(object):
    # process module
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='CutROI', training=False),
        dict(type='CutImage', training=False, window=(1000, 1000), step=(500, 500), order_index=False,
             is_keep_none=True)
    ]
    compose = Compose(train_pipeline)

    # data module
    img_dir = "/home/lifeng/undone-work/DefectNet/tools/data/tile/raw/tile_round1_testA_20201231/testA_imgs"
    test_file = "/home/lifeng/undone-work/dataset/detection/tile/annotations/instance_testA.json"
    save_file = "/home/lifeng/undone-work/DetCompetition/mmdet-v2/work_dirs/tile/baseline_cut_1000x1000/do_submit_testA.json"
    original_coco = COCO(test_file)
    label2name = {x['id']: x['name'] for x in original_coco.dataset['categories']}
    main_thread_lock = threading.Lock()
    save_results = []
    num_workers = 7
    process_cnt = 0

    # inference module
    device = 'cuda:0'
    config_file = '/home/lifeng/undone-work/DefectNet/configs/tile/baseline_model_2000x2000.py'
    checkpoint_file = '/data/liphone/detcomp/mmdet-v2/tile/baseline_cut_1000x1000/epoch_12.pth'
    model = init_detector(config_file, checkpoint_file, device=device)


def do_work(images, config):
    for image in tqdm(images):
        image['filename'] = image['file_name']
        results = {
            'img_prefix': config.img_dir,
            'img_info': image}
        results = config.compose(results)
        if results is None: results = []
        for i, result in enumerate(results):
            bbox_result = inference_detector(config.model, result['img'])
            # img = np.array(result['img'])
            for label, predicts in enumerate(bbox_result):
                for r in predicts:
                    # b = list(map(int, r[:4]))
                    # cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 3)
                    bbox = list(map(float, r[:4]))
                    if 'top_left' in result:
                        bbox = [bbox[0] + result['top_left'][0], bbox[1] + result['top_left'][1],
                                bbox[2] + result['top_left'][0], bbox[3] + result['top_left'][1]]
                    if 'roi_top_left' in result:
                        bbox = [bbox[0] + result['roi_top_left'][0], bbox[1] + result['roi_top_left'][1],
                                bbox[2] + result['roi_top_left'][0], bbox[3] + result['roi_top_left'][1]]
                    category_id, score = config.label2name[label + 1], r[4]
                    pred = {'name': str(image['filename']), 'category': int(category_id),
                            'bbox': bbox,
                            'score': float(score)}
                    config.save_results.append(pred)
            # plt.imshow(img)
            # plt.show()
            # cv2.imwrite("a.jpg", img)
        config.process_cnt += 1
        if config.process_cnt % 1 == 0 or config.process_cnt == len(images):
            print("process {}/{}...".format(config.process_cnt, len(images)))
        # for rst in config.save_results:
        #     img = cv2.imread(os.path.join(config.img_dir, rst['name']))
        #     b = list(map(int, rst['bbox'][:4]))
        #     cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 3)
        #     #plt.imshow(img)
        #     #plt.show()
        #      cv2.imwrite("a.jpg", img)
    return True


def main():
    config = CutConfig()
    if not os.path.exists(os.path.dirname(config.save_file)):
        os.makedirs(os.path.dirname(config.save_file))
    dataset = config.original_coco.dataset
    dataset['images'] = dataset['images']
    per_work_size = len(dataset['images']) // max(config.num_workers, 1)
    fetch, cnt = [], 0
    threads = []
    for i in range(config.num_workers):
        start = i * per_work_size
        end = start + per_work_size
        if (i + 1) == config.num_workers:
            end = len(dataset['images'])
        images = dataset['images'][start:end]
        cnt += len(images)
        threads.append(threading.Thread(target=do_work, args=(images, config)))
    assert cnt == len(dataset['images'])
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(config.save_file, "w") as fp:
        json.dump(config.save_results, fp, indent=4, ensure_ascii=False)
    print("process ok!")


if __name__ == '__main__':
    main()

import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
from sklearn.metrics import classification_report

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector, build_backbone


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results, result_times = [], []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            start_t = time.time()
            result = model(data['img'][0], return_loss=False, rescale=not show)
            result_times.append(time.time() - start_t)
            pred = torch.argmax(result, dim=1)
            results.append(int(pred))

        if show:
            model.module.show_result(data, result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, result_times


def have_defect(anns, images, threshold=0.05):
    if isinstance(anns, str):
        with open(anns) as fp:
            anns = json.load(fp)
    if isinstance(anns, dict):
        anns = anns['annotations']
    assert isinstance(anns, list)
    annotations = json_normalize(anns)
    # assert annotations.shape[0] == annotations[annotations['score'] > 0.05].shape[0]
    det_results = []
    for image in images:
        defect_num = 0
        if annotations.shape[0] > 0:
            ann = annotations[annotations['image_id'] == image['id']]
            for j in range(ann.shape[0]):
                a = ann.iloc[j]
                if 'score' in a:
                    if a['score'] > threshold and a['category_id'] > 0:
                        defect_num += 1
                else:
                    if a['category_id'] > 0:
                        defect_num += 1
        det_results.append(defect_num)
    assert len(det_results) == len(images)
    return det_results


def defect_eval(det_result, gt_result, result_times, threshold=0.05):
    pred_nums = det_result
    y_pred = [0 if x == 0 else 1 for x in pred_nums]
    true_nums = have_defect(gt_result, gt_result['images'], threshold)
    y_true = [0 if x == 0 else 1 for x in true_nums]
    assert len(y_pred) == len(y_true)

    find_ability_rpt = classification_report(y_true, y_pred, output_dict=False)
    find_ability = classification_report(y_true, y_pred, output_dict=True)
    defect_fps = [result_times[i] for i, x in enumerate(y_true) if x != 0]
    normal_fps = [result_times[i] for i, x in enumerate(y_true) if x == 0]

    assert len(defect_fps) + len(normal_fps) == len(result_times)
    return dict(
        log=dict(
            find_ability=find_ability_rpt, fps=np.mean(result_times),
            defect_fps=np.mean(defect_fps), normal_fps=np.mean(normal_fps),
        ),
        data=dict(
            find_ability=find_ability, fps=result_times,
            defect_fps=defect_fps, normal_fps=normal_fps,
        ),
    )


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument(
        '--config',
        default='../config_alcohol/cascade_rcnn_r50_fpn_1x/baseline.py',
        help='test config file path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file')
    parser.add_argument(
        '--json_out',
        default=None,
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        default=['bbox'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--imgs_per_gpu', type=int, default=1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(**kwargs):
    args = parse_args()
    for k, v in kwargs.items():
        args.__setattr__(k, v)

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    if isinstance(args.config, str):
        cfg = mmcv.Config.fromfile(args.config)
    else:
        cfg = args.config

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.val.test_mode = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    if args.mode == 'val':
        dataset = build_dataset(cfg.data.val)
    else:
        dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=args.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_backbone(cfg.model['backbone'])
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, result_times = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs, result_times = multi_gpu_test(model, data_loader, args.tmpdir,
                                               args.gpu_collect)

    rank, _ = get_dist_info()
    rpts = {}
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                rpts = coco_eval(result_file, eval_types, dataset.coco, classwise=True)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    rpts = coco_eval(result_files, eval_types, dataset.coco, classwise=True)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_, result_file)
                        rpts = coco_eval(result_files, eval_types, dataset.coco, classwise=True)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            rpts['bbox'] = dict(log={}, data={})
            defect_rpt = defect_eval(outputs, dataset.coco.dataset, result_times)
            rpts['bbox']['log']['defect_eval'] = defect_rpt['log']
            rpts['bbox']['data']['defect_eval'] = defect_rpt['data']
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)
    return rpts


if __name__ == '__main__':
    main()

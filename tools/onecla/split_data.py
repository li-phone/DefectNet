import pandas as pd
from tqdm import tqdm
import os
from utils import *


def split_chunk(raw_path, splits, balance=True):
    data_df = pd.read_csv(raw_path)
    keep_dfs = []
    if balance:
        column_names = list(data_df.columns)
        target_uniques = data_df[column_names[1]].unique()
        for x in target_uniques:
            keep_df = data_df[data_df[column_names[1]] == x]
            keep_dfs.append(keep_df)
    remaining_ratio = 1.0
    for s in splits:
        if balance:
            keep_df = pd.DataFrame()
            frac = s['ratio'] / remaining_ratio
            for i, x in enumerate(keep_dfs):
                tmp_df = x.sample(frac=frac)
                keep_dfs[i] = keep_dfs[i].drop(index=tmp_df.index)
                keep_df = pd.concat([keep_df, tmp_df])
            keep_df = keep_df.sample(frac=1, random_state=None)
            remaining_ratio -= s['ratio']
        else:
            keep_df = data_df.sample(frac=s['ratio'])
            data_df = data_df.drop(index=keep_df.index)
        keep_df.to_csv(s['ann_file'], index=False, header=True)


def main():
    # 预先定义环境
    dataset_cfg = import_module("cfg.py")
    cfg = import_module(dataset_cfg.dataset_cfg_path)
    mkdirs(cfg.data_root + 'annotations')

    split_chunk(cfg.dataset['raw_train_path'], cfg.dataset['train'], cfg.dataset['balance'])
    # split_chunk(cfg.raw_test_path, cfg.test, cfg.balance)
    print('split train data successfully!')


if __name__ == "__main__":
    main()

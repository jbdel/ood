import numpy as np
import torch
import operator
import os
import glob
import re


def get_metadata(name, df, index):
    d = {
        "index": index,
        "id": df.iloc[index, 0],
    }

    if name == 'skeletal-age':
        d.update({
            "sex": "M" if df.iloc[index, 2] else "F",
            "max_age": 228,
        })

    if name == 'mura':
        d.update({
            "bodypart": df.iloc[index, 2],
        })

    if name == 'mimix-crx':
        d.update({
            "view": df.iloc[index, 1],
        })

    return d


def get_label(name, df, index):
    if name == 'skeletal-age':
        label = df.iloc[index, 1] - 1  # there is no 0 month label
    elif name == 'mura':
        label = df.iloc[index, 1]
    elif name == 'retina':
        label = df.iloc[index, 1]
    elif name == 'mimic-crx':
        label = df.iloc[index, 2]
    else:
        raise NotImplementedError

    return np.array(label, dtype=np.long)


def compute_args(args):
    # Root
    args.root = {'skeletal-age': 'data/skeletal-age/',
                 'retina': 'data/retina/',
                 'mura': 'data/mura/',
                 'mimic-crx': 'data/mimic-crx'
                 }
    # Num classes
    if args.idd_name == 'mura':
        args.num_classes = 2
    elif args.idd_name == 'skeletal-age':
        args.num_classes = 228
    elif args.idd_name == 'retina':
        args.num_classes = 5
    elif args.idd_name == 'mimic-crx':
        args.num_classes = 2
    else:
        raise NotImplementedError

    # Early stop comparison
    if args.early_stop_metric == 'fpr_at_95_tpr':
        args.early_stop_operator = min
        args.early_stop_start = np.inf

    elif args.early_stop_metric == 'auroc':
        args.early_stop_operator = max
        args.early_stop_start = -np.inf

    elif args.early_stop_metric == 'accuracy':
        args.early_stop_operator = max
        args.early_stop_start = -np.inf
    else:
        raise NotImplementedError

    return args


def save_checkpoint(checkpoints_folder, state_dict, keep_n_best=5):
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoints = glob.glob(f'{checkpoints_folder}/model_*.pth')

    if len(checkpoints) == keep_n_best:
        # Need to delete worst checkpoint
        metrics = [float(re.search('model_(.*).pth', c).group(1)) for c in checkpoints]
        if state_dict["args"].early_stop_operator.__name__ == 'min':
            worst_metric = max(metrics)
        else:
            worst_metric = min(metrics)
        os.remove(f'{checkpoints_folder}/model_{worst_metric}.pth')

    checkpoint_filename = f'{checkpoints_folder}/model_{state_dict["best_early_stop_value"]}.pth'
    torch.save(state_dict, checkpoint_filename)
    return None

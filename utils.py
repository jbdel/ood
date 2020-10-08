import numpy as np


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
    return d


def get_label(name, df, index):
    if name == 'skeletal-age':
        label = df.iloc[index, 1] - 1 # there is no 0 month label
        male = df.iloc[index, 2]
        max_age = 228
        if male:
            label += max_age
    elif name == 'mura':
        label = df.iloc[index, 1]
    else:
        raise NotImplementedError

    return np.array(label)


def compute_args(args):
    args.root = {'skeletal-age': 'data/skeletal_age/',
                 'retina': 'data/retina/',
                 'mura': 'data/mura/',
                 'mimic-crx': 'data/mimic-crx'
                 }
    if args.idd_name == 'mura':
        args.num_classes = 2
    elif args.idd_name == 'skeletal-age':
        args.num_classes = 228*2
    elif args.idd_name == 'retina':
        args.num_classes = 5
    elif args.idd_name == 'mimic-crx':
        args.num_classes = 2
    else:
        raise NotImplementedError
    return args

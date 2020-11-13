import torch
import argparse

parser = argparse.ArgumentParser()
# Experiment
parser.add_argument("--ckpt", type=str, default="default")
args = parser.parse_args()

checkpoint = torch.load(args.ckpt)
for k, v in checkpoint.items():
    if k in ['ood_metric_strs', 'ind_metric_str']:
        print(k, v)



import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from metrics import calc_metrics
from torch.utils.data import DataLoader
from dataset import LarsonDataset
from models import (
    DeVriesLarsonModelConfig,
    LarsonModelConfig,
)
from collections import defaultdict
import utils


def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--idd_name", default="skeletal-age", choices=['retina', 'skeletal-age', 'mura', 'mimic-crx'])
    parser.add_argument("--ood_name", type=str, nargs='+', default=['mura'])
    parser.add_argument("--model", default="default", choices=['devries', 'default'])
    parser.add_argument("--network", type=str, default="resnet")
    # Hyper params
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--hint_rate", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lmbda", type=float, default=0.1)
    # Training params
    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument('--losses', nargs='+', default=["boneage_mad", "accuracy"])
    parser.add_argument('--early_stop_metric', type=str, default="fpr_at_95_tpr")
    parser.add_argument('--early_stop', type=int, default=20)
    # Misc
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--load_memory", type=bool, default=False, help="Load images into CPU")

    args = parser.parse_args()
    args = utils.compute_args(args)

    # Create dataloader according to experiments
    loader_args = {'name': args.idd_name,
                   'mode': 'idd',
                   'root_dir': args.root[args.idd_name],
                   'csv_file': 'train.csv',
                   'load_memory': args.load_memory}

    train_loader = DataLoader(LarsonDataset(**loader_args),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    test_true_loader = DataLoader(LarsonDataset(**dict(loader_args, **{'csv_file': 'test.csv'})),
                                  batch_size=16,
                                  shuffle=False,
                                  num_workers=4)

    test_false_loaders = {}
    for ood_name in args.ood_name:
        test_false_loaders[ood_name] = DataLoader(LarsonDataset(**{'name': ood_name,
                                                                   'mode': 'ood',
                                                                   'root_dir': args.root[ood_name],
                                                                   'csv_file': 'test.csv',
                                                                   'load_memory': False
                                                                   }),
                                                  batch_size=16,
                                                  shuffle=False,
                                                  num_workers=4)

    if args.model == "devries":
        model_config = DeVriesLarsonModelConfig(args=args,
                                                hint_rate=args.hint_rate,
                                                lmbda=args.lmbda)
    else:
        model_config = LarsonModelConfig(args=args)
    net = model_config.net

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(net)
        net = model.module
    print("Model", args.model, "Network", args.network,
          "\nTotal number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

    init_epoch = 0

    # Restores settings
    checkpoints_folder = f"checkpoints/{args.experiment_name}"
    checkpoint_filename = f"{checkpoints_folder}/{args.checkpoint}"
    if os.path.isfile(checkpoint_filename):
        print(f"loading: {checkpoint_filename}")
        checkpoint = torch.load(checkpoint_filename)
        init_epoch = checkpoint["init_epoch"]
        net.load_state_dict(checkpoint["net"])
        model_config.optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_scheduler:
            model_config.scheduler.load_state_dict(checkpoint["scheduler"])

    early_stop = 0
    best_early_stop_value = 0
    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        train_start = time.time()

        # Train phase
        net.train()
        for train_iter, sample in enumerate(train_loader, 0):
            inputs_batch, label_batch, _ = sample
            # Reassigns inputs_batch and label_batch to cuda
            cuda_inputs_batch = inputs_batch.cuda()
            cuda_label_batch = label_batch.cuda()

            model_config.optimizer.zero_grad()
            output_batches = net(cuda_inputs_batch)
            total_loss, task_loss, confidence_loss = model_config.criterion(output_batches, cuda_label_batch)

            print(
                "\r[Epoch {}][Step {}/{}] Loss: {:.2f} [Task: {:.2f}, Confidence: {:.2f}], lambda: {:.2f}, {:.2f} m remaining".format(
                    epoch + 1,
                    train_iter,
                    int(len(train_loader.dataset) / args.batch_size),
                    total_loss,
                    task_loss,
                    confidence_loss,
                    model_config.criterion.lmbda,
                    ((time.time() - train_start) / (train_iter + 1)) * (
                            (len(train_loader.dataset) / args.batch_size) - train_iter) / 60,
                ), end='          ')
            total_loss.backward()
            model_config.optimizer.step()
            break

        # Eval phase
        net.eval()

        def evaluate(data_loader, mode="confidence", ood=False):
            out_conf = []
            idd_metrics = defaultdict(list)

            for test_iter, sample in enumerate(data_loader, 0):
                inputs_batch, label_batch, metadata_batch = sample

                # Reassigns inputs_batch and label_batch to cuda
                cuda_inputs_batch = inputs_batch.cuda()
                task_sore, confidence = net(cuda_inputs_batch)

                # manage in domain metric
                if not ood:
                    metric = model_config.task_metric(task_sore,
                                                      label_batch,
                                                      metadata_batch)
                    for k, v in metric.items():
                        idd_metrics[k].extend(v)

                # save confidence of ood metrics
                confidence = torch.sigmoid(confidence)
                confidence = confidence.data.cpu().numpy()
                out_conf.append(confidence)

            out_conf = np.concatenate(out_conf)

            if not ood:
                for k, v in idd_metrics.items():
                    idd_metrics[k] = np.mean(v)

            return out_conf, idd_metrics

        # In domain evaluation
        ind_confs, ind_metric = evaluate(test_true_loader)
        ind_labels = np.ones(ind_confs.shape[0])

        # Out of domain evaluation
        early_stop_metrics = 0
        for ood_name, test_false_loader in test_false_loaders.items():
            ood_confs, _ = evaluate(test_false_loader, ood=True)
            ood_labels = np.zeros(ood_confs.shape[0])

            labels = np.concatenate([ind_labels, ood_labels])
            scores = np.concatenate([ind_confs, ood_confs])

            error_metrics = calc_metrics(scores, labels)
            error_metrics.update(ind_metric)

            error_metric_str = ", ".join([f"{k}: {v}" for k, v in error_metrics.items()])
            print(f"\rEpoch: {epoch}, OOD Name: {ood_name}, {error_metric_str}\n")
            early_stop_metrics += error_metrics[args.early_stop_metric]

        early_stop_metrics = early_stop_metrics / len(test_false_loaders)
        early_stop += 1
        if early_stop_metrics > best_early_stop_value:
            os.makedirs(checkpoints_folder, exist_ok=True)
            checkpoint_filename = f"{checkpoints_folder}/model.pth"
            torch.save({
                "init_epoch": epoch + 1,
                "net": net.state_dict(),
                "optimizer": model_config.optimizer.state_dict(),
                "scheduler": model_config.scheduler.state_dict() if args.use_scheduler else None,
            }, checkpoint_filename)
            open(f"{checkpoints_folder}/best.txt", 'w+').write(error_metric_str)
            early_stop = 0

        if args.use_scheduler:
            model_config.scheduler.step()
        if early_stop == args.early_stop:
            print("early_stop reached")
            return


if __name__ == "__main__":
    main()

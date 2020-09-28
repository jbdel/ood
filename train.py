import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from metrics import calc_metrics
from torch.utils.data import DataLoader
from dataset import LarsonDataset
from models import (
    DeVriesLarsonModelConfig,
    LarsonModelConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="",
                        help="Path to write checkpoints, results, etc., relative to 'checkpoints/'")
    parser.add_argument("--checkpoint", default="",
                        help="Path to checkpoint to load, relative to 'checkpoints/{args.experiment_name}/'. If empty, starts from scratch.")
    parser.add_argument("--overfit", default=False,
                        help="Whether or not to overfit on the test dataset. Turns off random transforms.")
    parser.add_argument("--model", default="default",
                        help="Model to train (Options: devries|default).")
    parser.add_argument("--num_epochs", type=int,
                        help="Number of epochs to train.")
    parser.add_argument("--hint_rate", type=float)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lmbda", type=float)
    parser.add_argument("--network", type=str, default="resnet")
    parser.add_argument("--use_scheduler", type=bool, default=False)
    args = parser.parse_args()

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
    print("Model", args.model, "Network", args.network, "\nTotal number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

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

    # Defines train and test datasets and loaders
    train_dataset_kwargs = {
        "csv_file": "data/train.csv",
        "root_dir": "data",
        "input_transforms": model_config.train_input_transforms,
        "label_transforms": model_config.train_label_transforms,
    }
    test_true_dataset_kwargs = {
        "csv_file": "data/test_true.csv",
        "root_dir": "data",
        "input_transforms": model_config.test_input_transforms,
        "label_transforms": model_config.test_label_transforms,
    }

    test_false_dataset_kwargs = {
        "csv_file": "data/test_false.csv",
        "root_dir": "data",
        "input_transforms": model_config.test_input_transforms,
        "label_transforms": model_config.test_label_transforms,
    }

    train_loader = DataLoader(LarsonDataset(**train_dataset_kwargs),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    test_true_loader = DataLoader(LarsonDataset(**test_true_dataset_kwargs),
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=4)

    test_false_loader = DataLoader(LarsonDataset(**test_false_dataset_kwargs),
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=4)

    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        train_start = time.time()

        # Train phase
        net.train()
        for train_iter, sample in enumerate(train_loader, 0):
            inputs_batch, label_batch, _ = sample

            # Reassigns inputs_batch and label_batch to cuda
            cuda_inputs_batch = list(map(lambda x: x.cuda(), inputs_batch))
            cuda_label_batch = label_batch.cuda()

            model_config.optimizer.zero_grad()
            output_batches = net(*cuda_inputs_batch)
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

        # Eval phase
        net.eval()

        def evaluate(data_loader, mode="confidence"):
            out_conf = []
            out_mad = []
            for test_iter, sample in enumerate(data_loader, 0):
                inputs_batch, label_batch, metadata_batch = sample

                # Reassigns inputs_batch and label_batch to cuda
                cuda_inputs_batch = list(map(lambda x: x.cuda(), inputs_batch))
                task_sore, confidence = net(*cuda_inputs_batch)

                mad = model_config.task_metric(task_sore,
                                               metadata_batch)
                confidence = torch.sigmoid(confidence)
                confidence = confidence.data.cpu().numpy()
                out_conf.append(confidence)
                out_mad.extend(mad)

            out_conf = np.concatenate(out_conf)
            out_mad = np.mean(out_mad)

            return out_conf, out_mad

        ind_confs, ind_mad = evaluate(test_true_loader)
        ind_labels = np.ones(ind_confs.shape[0])

        ood_confs, _ = evaluate(test_false_loader)
        ood_labels = np.zeros(ood_confs.shape[0])

        labels = np.concatenate([ind_labels, ood_labels])
        scores = np.concatenate([ind_confs, ood_confs])

        error_metrics = calc_metrics(scores, labels)
        error_metric_str = ", ".join([f"{k}: {v}" for k, v in error_metrics.items()])
        print(f"\rEpoch: {epoch}, {error_metric_str}\n")

        os.makedirs(checkpoints_folder, exist_ok=True)
        checkpoint_filename = f"{checkpoints_folder}/model.pth"
        torch.save({
            "init_epoch": epoch + 1,
            "net": net.state_dict(),
            "optimizer": model_config.optimizer.state_dict(),
            "scheduler": model_config.scheduler.state_dict() if args.use_scheduler else None,
        }, checkpoint_filename)

        if args.use_scheduler:
            model_config.scheduler.step()

if __name__ == "__main__":
    main()

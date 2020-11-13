import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from metrics import calc_metrics
from torch.utils.data import DataLoader, ConcatDataset
from dataset import LarsonDataset
from models import (
    DeVriesLarsonModelConfig,
)
from collections import defaultdict
import utils
from plot import Plot
from metrics import plot_metrics, plot_classification
from confidence import get_confidence
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--experiment_name", default="default")
    parser.add_argument("--idd_name", default="skeletal-age")
    parser.add_argument("--mode", type=str, default='devries',
                        choices=['baseline', 'devries', 'devries_odin', 'energy', 'oe'])
    parser.add_argument("--outlier_name", type=str, nargs='+', default=['mura', 'mimic-crx'])
    parser.add_argument("--ood_name", type=str, nargs='+', default=['retina'])
    parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')

    parser.add_argument("--network", type=str, default="resnet")
    # Hyper params
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--hint_rate", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--lmbda", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--m_in', type=float, default=-25.,
                        help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7.,
                        help='margin for out-distribution; below this value will be penalized')
    # Training params
    parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument('--losses', nargs='+', default=["boneage_mad", "accuracy"])
    parser.add_argument('--early_stop_metric', type=str, default="fpr_at_95_tpr")
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--eval_start', type=int, default=1)
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

    outlier_set = ConcatDataset([LarsonDataset(**{'name': outlier_name,
                                                  'mode': 'idd',
                                                  'root_dir': args.root[outlier_name],
                                                  'csv_file': 'train.csv',
                                                  'load_memory': False
                                                  }) for outlier_name in args.outlier_name])

    assert len(outlier_set) >= len(train_loader.dataset)

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

    model_config = DeVriesLarsonModelConfig(args=args,
                                            hint_rate=args.hint_rate,
                                            lmbda=args.lmbda,
                                            beta=args.beta)
    net = model_config.net
    loss_plots = Plot(idd_name=args.idd_name, early_stop_metric=args.early_stop_metric)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    print("Network", args.network, 'mode', args.mode,
          "\nLambda", args.lmbda, "beta", args.beta, "hint_rate", args.hint_rate,
          "\nTotal number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

    init_epoch = 0
    checkpoints_folder = f"checkpoints/{args.experiment_name}"
    early_stop = 0

    best_early_stop_value = args.early_stop_start  # -inf or +inf
    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        train_start = time.time()
        # Train phases
        net.train()

        # Shuffling outlier set
        outlier_loader = DataLoader(outlier_set,
                                    batch_size=16,
                                    shuffle=True,
                                    num_workers=4)

        for train_iter, (in_set, out_set) in enumerate(zip(train_loader,outlier_loader)):

            data = torch.cat((in_set[0], out_set[0]), dim=0)
            target = in_set[1]

            data, target = data.cuda(), target.cuda()

            # forward
            pred, _ = net(data)

            # backward
            model_config.optimizer.zero_grad()

            task_loss = F.cross_entropy(pred[:len(in_set[0])], target)
            # cross-entropy from softmax distribution to uniform distribution
            if args.mode == 'energy':
                Ec_out = -torch.logsumexp(pred[len(in_set[0]):], dim=1)
                Ec_in = -torch.logsumexp(pred[:len(in_set[0])], dim=1)
                oe_loss = 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                          2).mean())
            elif args.mode == 'oe':
                oe_loss = args.beta * -(pred[len(in_set[0]):].mean(1) - torch.logsumexp(pred[len(in_set[0]):], dim=1)).mean()
            else:
                raise NotImplementedError

            total_loss = task_loss + oe_loss
            total_loss.backward()
            model_config.optimizer.step()

            print(
                "\r[Epoch {}][Step {}/{}] Loss: {:.2f} [Task: {:.2f}, Energy: {:.2f}], Lr: {:.2e}, ES: {}, {:.2f} m remaining".format(
                    epoch + 1,
                    train_iter,
                    int(len(train_loader.dataset) / args.batch_size),
                    total_loss.cpu().data.numpy(),
                    task_loss.cpu().data.numpy(),
                    oe_loss.cpu().data.numpy(),
                    *[group['lr'] for group in model_config.optimizer.param_groups],
                    early_stop,
                    ((time.time() - train_start) / (train_iter + 1)) * (
                            (len(train_loader.dataset) / args.batch_size) - train_iter) / 60,
                ), end='          ')

        # Eval phase
        if epoch + 1 >= args.eval_start:
            net.eval()

            def evaluate(data_loader, mode="confidence", idd=False):
                confidences = []
                idd_metrics = defaultdict(list)

                for test_iter, sample in enumerate(data_loader, 0):
                    images, labels, _ = sample

                    # Reassigns inputs_batch and label_batch to cuda
                    pred, confidence = net(images.cuda())
                    labels = labels.data.cpu().numpy()

                    # manage in domain metric
                    if idd:
                        loss_dict = {}
                        task_predictions = torch.argmax(pred, dim=-1).data.cpu().numpy()
                        if "accuracy" in args.losses:
                            loss_dict['accuracy'] = list(task_predictions == labels)
                        elif "boneage_mad" in args.losses:
                            loss_dict['boneage_mad'] = list(abs(task_predictions - labels))
                        else:
                            raise NotImplementedError

                        for k, v in loss_dict.items():
                            idd_metrics[k].extend(v)

                    # Get confidence in prediction
                    confidences.extend(get_confidence(net, images, pred, confidence, args))

                confidences = np.array(confidences)
                if idd:
                    # Plot accuracy
                    if 'accuracy' in idd_metrics:
                        plot_classification(idd_metrics['accuracy'],
                                            confidences,
                                            checkpoints_folder,
                                            name=str(args.idd_name)
                                                 + ' - '
                                                 + str(round(float(np.mean(idd_metrics['accuracy'])), 4))
                                                 + ' - '
                                                 + 'Epoch ' + str(
                                                epoch + 1) + '.jpg')

                    # Average metric over valset 
                    for k, v in idd_metrics.items():
                        idd_metrics[k] = round(float(np.mean(v)), 4)

                return confidences, idd_metrics

            # In domain evaluation
            ind_confs, ind_metrics = evaluate(test_true_loader, idd=True)
            ind_labels = np.ones(ind_confs.shape[0])

            ind_metrics["IDD name"] = args.idd_name
            print(str(ind_metrics))

            # Out of domain evaluation
            early_stop_metric_value = 0
            ood_metric_dicts = []
            for ood_name, test_false_loader in test_false_loaders.items():
                ood_confs, _ = evaluate(test_false_loader, idd=False)
                ood_labels = np.zeros(ood_confs.shape[0])

                labels = np.concatenate([ind_labels, ood_labels])
                scores = np.concatenate([ind_confs, ood_confs])

                ood_metrics = calc_metrics(scores, labels)
                ood_metrics['OOD Name'] = ood_name
                print(str(ood_metrics))
                ood_metric_dicts.append(ood_metrics)

                # Plot metrics
                plot_metrics(scores, labels, ind_confs, ood_confs, checkpoints_folder, name=str(ood_name)
                                                                                            + ' - '
                                                                                            + 'Epoch ' + str(epoch + 1))

                # fetch early stop value (might be a iid metric)
                early_stop_metric_value += {**ind_metrics, **ood_metrics}[args.early_stop_metric]

            early_stop_metric_value = early_stop_metric_value / len(test_false_loaders)
            early_stop += 1
            loss_plots.update(epoch + 1, ind_metrics, ood_metric_dicts)
            # Save model + early stop
            # Early_stop_operator is min or max
            if args.early_stop_operator(early_stop_metric_value, best_early_stop_value) != best_early_stop_value:
                early_stop = 0
                best_early_stop_value = early_stop_metric_value
                utils.save_checkpoint(checkpoints_folder,
                                      {
                                          "init_epoch": epoch + 1,
                                          "net": net.state_dict(),
                                          "optimizer": model_config.optimizer.state_dict(),
                                          "scheduler": model_config.scheduler.state_dict() if args.use_scheduler else None,
                                          "ood_metrics": ood_metric_dicts,
                                          "ind_metrics": ind_metrics,
                                          "best_early_stop_value": best_early_stop_value,
                                          "args": args,
                                      },
                                      keep_n_best=1)

                print('Early stop metric ' + str(args.early_stop_metric) + ' beaten. Now ' + str(best_early_stop_value))

        if args.use_scheduler:
            model_config.scheduler.step(ind_metrics['accuracy'])
        if early_stop == args.early_stop:
            loss_plots.draw(checkpoints_folder)
            print("early_stop reached")
            break

    loss_plots.draw(checkpoints_folder)
    print('Done')
    return


if __name__ == "__main__":
    main()

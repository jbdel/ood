import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from metrics import calc_metrics
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.datasets as dset
from dataset import LarsonDataset
from models import (
    DeVriesLarsonModelConfig,
)
from collections import defaultdict
import utils
from plot import Plot
from metrics import plot_metrics, plot_classification
from confidence import get_confidence


def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--experiment_name", default="default")
    parser.add_argument("--idd_name", default="skeletal-age")
    parser.add_argument("--mode", type=str, default='devries',
                        choices=['baseline', 'devries', 'devries_odin', 'energy', 'oe'])
    parser.add_argument("--ood_name", type=str, nargs='+', default=['retina', 'mura', 'mimic-crx'])
    parser.add_argument("--network", type=str, default="resnet")
    # Hyper params
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--hint_rate", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--lmbda", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)

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
    import torchvision.transforms as trn
    train_data = dset.MNIST('MNIST', train=True, transform=trn.ToTensor(),download=True,)
    test_data = dset.MNIST('MNIST', train=False, transform=trn.ToTensor(),download=True,)
    num_classes = 10


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_true_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    print(len(train_loader))
    print(len(test_true_loader))
    test_false_loaders = {}

    # /////////////// gaussian Noise ///////////////


    dummy_targets = torch.ones(5000)
    ood_data = torch.from_numpy(
        np.clip(np.random.normal(size=(5000, 1, 28, 28),
                                 loc=0.5, scale=0.5).astype(np.float32), 0, 1))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=16, shuffle=True)
    test_false_loaders["gaussian"] = ood_loader

    # /////////////// Bernoulli Noise ///////////////

    dummy_targets = torch.ones(5000)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(5000, 1, 28, 28)).astype(np.float32))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=16, shuffle=True)

    test_false_loaders["Bernoulli"] = ood_loader

    # /////////////// CIFAR data ///////////////

    ood_data = dset.CIFAR10(
        'cifar', train=False, download=True,
        transform=trn.Compose([trn.Resize(28),
                               trn.Lambda(lambda x: x.convert('L', (0.2989, 0.5870, 0.1140, 0))),
                               trn.ToTensor()]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=16, shuffle=True,
                                             num_workers=4, pin_memory=True)
    test_false_loaders["CIFAR"] = ood_loader

    model_config = DeVriesLarsonModelConfig(args=args,
                                            hint_rate=args.hint_rate,
                                            lmbda=args.lmbda,
                                            beta=args.beta)

    def gelu(x):
        return torch.sigmoid(1.702 * x) * x
        # return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
            self.fc3 = nn.Linear(50, 1)

        def forward(self, x):
            import torch.nn.functional as F

            x = gelu(F.max_pool2d(self.conv1(x), 2))
            x = gelu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = gelu(self.fc1(x))
            # x = F.dropout(x)
            return self.fc2(x), self.fc3(x)

    net = ConvNet().cuda()
    import torch.optim as optim

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
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
        for train_iter, (data, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            data, labels = data.cuda(), labels.cuda()
            output_batches = net(data.cuda())
            total_loss, task_loss, confidence_loss = model_config.criterion(output_batches, labels.cuda())
            total_loss.backward()
            optimizer.step()

            print(
                "\r[Epoch {}][Step {}/{}] Loss: {:.2f} [Task: {:.2f}, Confidence: {:.2f}, lambda: {:.2f}], Lr: {:.2e}, ES: {}, {:.2f} m remaining".format(
                    epoch + 1,
                    train_iter,
                    int(len(train_loader.dataset) / args.batch_size),
                    total_loss.cpu().data.numpy(),
                    task_loss.cpu().data.numpy(),
                    confidence_loss.cpu().data.numpy(),
                    model_config.criterion.lmbda,
                    *[group['lr'] for group in optimizer.param_groups],
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

                for test_iter, (data, labels) in enumerate(data_loader, 0):
                    data = data.view(-1, 1, 28, 28).cuda()

                    # Reassigns inputs_batch and label_batch to cuda
                    pred, confidence = net(data.cuda())
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
                    confidences.extend(get_confidence(net, data, pred, confidence, args))

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
                                          "optimizer": optimizer.state_dict(),
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

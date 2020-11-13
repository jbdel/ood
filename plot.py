from collections import defaultdict
import os
import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, idd_name, early_stop_metric):
        self.idd_name = idd_name
        self.early_stop_metric = early_stop_metric
        self.metrics_per_set = defaultdict(lambda: defaultdict(list))  # dict of dict of list
        self.metrics_ood = defaultdict(list)
        self.metrics_idd = defaultdict(list)
        self.epochs = []

    def update(self, epoch, ind_metric, ood_metric_dicts):
        self.epochs.append(epoch)
        for k, v in ind_metric.items():
            self.metrics_idd[k].append(v)

        # ood_metric_dicts is a list of dictionary (one ood dict per ood test set)
        ood_metrics = list(ood_metric_dicts[0].keys())
        ood_metrics.remove('OOD Name')
        # for each OOD metric
        for metric in ood_metrics:
            m = []
            # Iter over ood dataset and get metric
            for ood_dict in ood_metric_dicts:
                ood_name = ood_dict['OOD Name']
                value = ood_dict[metric]
                self.metrics_per_set[ood_name][metric].append(value)
                m.append(value)
            # Also record average ood metric amongst sets
            self.metrics_ood[metric].append(np.mean(m))

    def draw(self, checkpoints_folder):
        out_dir = os.path.join(checkpoints_folder, 'plots')
        os.makedirs(out_dir, exist_ok=True)

        # Plot per metrics (on average amongst ood test set for ood)
        for d in [self.metrics_ood, self.metrics_idd]:
            for k, v in d.items():
                plt.plot(self.epochs, v, label=str(k))
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.savefig(os.path.join(out_dir, str(k) + '.png'))
                plt.clf()
                plt.close('all')

        # save per category metrics
        metrics = list(self.metrics_ood.keys())
        for ood_set, metric_d in self.metrics_per_set.items():
            example_metric = metric_d
            for i, metric in enumerate(metrics):
                plt.figure(i+1)
                plt.plot(self.epochs, metric_d[metric], label=str(ood_set))

        for i, metric in enumerate(metrics):
            plt.figure(i+1)
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title(str(metric))
            plt.savefig(os.path.join(out_dir, 'sets_' + str(metric) + '.png'))

# x = np.linspace(0, 2, 100)
#
# plt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.
# plt.plot(x, x ** 2, label='quadratic')  # etc.
# plt.plot(x, x ** 3, label='cubic')
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.title("Simple Plot")
# plt.legend()

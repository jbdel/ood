import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torchvision.models.resnet import resnet50
from torch.optim.lr_scheduler import MultiStepLR
from densenet3 import DenseNet3
from torchvision.transforms import Compose
import torch.nn.functional as F


class LarsonModelConfig(object):

    def __init__(self, args):
        self._args = args
        self._net = None
        self._label = None
        self._criterion = None
        self._optimizer = None
        self._scheduler = None
        self._error_metrics = None

    @property
    def optimizer(self):
        if self._optimizer is None:
            # self._optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1,
            #                                   momentum=0.9, nesterov=True, weight_decay=1e-4)
            self._optimizer = optim.Adam(self.net.parameters(), lr=self._args.lr)
        return self._optimizer

    @property
    def scheduler(self):
        if self._args.use_scheduler:
            self._scheduler = MultiStepLR(self._optimizer, milestones=[150, 225], gamma=0.1)
            return self._scheduler
        return None

    @property
    def net(self):
        """
    Returns an instance of nn.Module.
    """
        if self._net is None:
            self._net = self.Network(self._args)

        return self._net

    @property
    def criterion(self):
        self._criterion = nn.CrossEntropyLoss
        return self._criterion

    class Network(nn.Module):

        def __init__(self, args):
            super().__init__()
            if args.network == "resnet":
                self._net = resnet50(pretrained=True).cuda()
                self._net.fc = nn.Linear(2048, args.num_classes).cuda()
            if args.network == "densenet3":
                self._net = DenseNet3(depth=100, num_classes=args.num_classes, growth_rate=12, reduction=0.5).cuda()

        def forward(self, inputs):
            return self._net(inputs)


class DeVriesLarsonModelConfig(LarsonModelConfig):

    def __init__(self, args, hint_rate=0.0, lmbda=1.0):
        super().__init__(args)
        self.hint_rate = hint_rate
        self.lmbda = lmbda

    @property
    def criterion(self):
        self._criterion = self.Criterion(hint_rate=self.hint_rate,
                                         lmbda=self.lmbda)

        return self._criterion

    def task_metric(self, task_score_batch, label_batch, metadata_batch):
        loss_dict = {}
        task_targets = label_batch.numpy()

        if "boneage_mad" in self._args.losses:
            scores_batch = []
            for batch_iter, item in enumerate(task_score_batch, 0):
                sex = metadata_batch['sex'][batch_iter]
                max_age = metadata_batch['max_age'][batch_iter]
                if sex == "F":
                    scores = item[:max_age]
                else:
                    scores = item[max_age:]

                scores_batch.append(torch.argmax(scores, dim=-1))
            task_predictions = torch.stack(scores_batch).cpu().numpy()
            mad = abs(task_predictions - task_targets)
            loss_dict['boneage_mad'] = list(mad)

        if "accuracy" in self._args.losses:
            task_predictions = torch.argmax(task_score_batch, dim=-1).cpu().numpy()
            loss_dict['accuracy'] = list(task_predictions == task_targets)

        return loss_dict

    class Network(nn.Module):
        def __init__(self, args):
            super().__init__()
            if args.network == "resnet":
                self._net = resnet50(pretrained=True).cuda()
                self._net.fc = nn.Linear(2048, args.num_classes + 1).cuda()
            if args.network == "densenet3":
                self._net = DenseNet3(depth=100, num_classes=args.num_classes + 1, growth_rate=12, reduction=0.5).cuda()

        def forward(self, inputs):
            scores = self._net(inputs)
            confidence_score = scores[:, 0]
            task_scores = scores[:, 1:]

            return task_scores, confidence_score

    class Criterion(nn.Module):

        def __init__(self, hint_rate, lmbda, beta=0.3):
            """
      Inputs:
      - hint_rate: A Python float; hint_rate=1.0 means the model asks for all
        hints and hint_rate=0.0 means the model asks for no hints
      - lmbda: A Python float that represents the relative weighting between
        task loss and confidence loss
      """
            super().__init__()
            self.hint_rate = hint_rate
            self.lmbda = lmbda
            self.beta = 0.3

        def forward(self, inputs, target):
            task_scores, confidence_score = inputs
            task_probs = F.softmax(task_scores, dim=-1)
            confidence_prob = torch.sigmoid(confidence_score)
            _, num_classes = task_scores.size()

            one_hot_target = nn.functional.one_hot(target, num_classes=num_classes)
            # Make sure we don't have any numerical instability
            eps = 1e-12
            task_probs = torch.clamp(task_probs, 0. + eps, 1. - eps)
            confidence_prob = torch.clamp(confidence_prob, 0. + eps, 1. - eps)

            b = torch.bernoulli(torch.empty(confidence_prob.shape).uniform_(0, 1)).cuda()
            conf = confidence_prob * b + (1 - b)
            pred_new = task_probs * conf.unsqueeze(-1) + one_hot_target * (1 - conf.unsqueeze(-1))
            pred_new = torch.log(pred_new)

            xentropy_loss = nn.NLLLoss()(pred_new, target)
            confidence_loss = torch.mean(-torch.log(confidence_prob))

            total_loss = xentropy_loss + (self.lmbda * confidence_loss)

            if self.beta > confidence_loss.data:
                self.lmbda = self.lmbda / 1.01
            elif self.beta <= confidence_loss.data:
                self.lmbda = self.lmbda / 0.99

            return total_loss, xentropy_loss.cpu().data.numpy(), confidence_loss.cpu().data.numpy()

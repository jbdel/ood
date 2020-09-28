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

from transforms import (
    RepeatGrayscaleChannels,
    Clahe,
    RandomSquareCrop,
    RandomHorizontalFlip,
    Transpose,
    ToTensor,
    ToFloat,
    ToLong,
)


class ModelConfig(object):

    def __init__(self, args):
        self._args = args
        self._net = None
        self._label = None
        self._criterion = None
        self._optimizer = None
        self._scheduler = None
        self._train_input_transforms = None
        self._test_input_transforms = None
        self._train_label_transforms = None
        self._test_label_transforms = None
        self._error_metrics = None

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1,
                                              momentum=0.9, nesterov=True, weight_decay=1e-4)
        return self._optimizer

    @property
    def scheduler(self):
        if self._args.use_scheduler:
            self._scheduler = MultiStepLR(self._optimizer, milestones=[150, 225], gamma=0.1)
            return self._scheduler
        return None

    @property
    def net(self):
        raise NotImplementedError

    @property
    def criterion(self):
        raise NotImplementedError

    @property
    def train_input_transforms(self):
        raise NotImplementedError

    @property
    def test_input_transforms(self):
        raise NotImplementedError

    @property
    def train_label_transforms(self):
        raise NotImplementedError

    @property
    def test_label_transforms(self):
        raise NotImplementedError


class LarsonModelConfig(ModelConfig):

    def __init__(self, args, age_range=(0, 240, 1), sex=True):
        """
    Inputs:
    - age_range: A Python 2-tuple or 3-tuple passed to range, that represents
      the predicted age classes.
    - sex: A Python bool that represents whether or not to separate predicted
      age classes by sex.
    """
        super().__init__(args)
        self.age_range = age_range
        self.sex = sex
        self._num_ages = None
        self._num_classes = None

    @property
    def num_ages(self):
        if self._num_ages is None:
            self._num_ages = len(list(range(*self.age_range)))

        return self._num_ages

    @property
    def num_classes(self):
        if self._num_classes is None:
            self._num_classes = self.num_ages * 2 if self.sex else self.num_ages

        return self._num_classes

    @property
    def net(self):
        """
    Returns an instance of nn.Module.
    """
        if self._net is None:
            self._net = self.Network(self.num_classes, self._args)

        return self._net

    @property
    def criterion(self):
        """
    In general, should return a criterion that expects:
    - One batch per output of self.net.__call__
    - One batch of the label yielded by the dataset, according to the label
      transform

    For this particular model, self.net.__call__ returns a single output
    consisting of pre-softmax scores for n categories, and self.label returns
    a numpy array with shape (1,) representing the desired index into n
    categories.
    """
        if self._criterion is None:
            self._criterion = self.Criterion()

        return self._criterion

    @property
    def train_input_transforms(self):
        """
    Returns a list of Compose objects, one per input expected by self.net.

    Each Compose object expects a dict with keys:
    - image_arr
    - skeletal_age
    - male
    and outputs the input in the format expected by self.net for that input.
    """
        if self._train_input_transforms is None:
            self._train_input_transforms = [
                Compose([
                    lambda x: x["image_arr"],
                    RepeatGrayscaleChannels(3),
                    Clahe(),  # noop at the moment
                    RandomSquareCrop((224, 224)),
                    RandomHorizontalFlip(),
                    Transpose(),
                    ToTensor(),
                    ToFloat(),
                ]),
            ]

        return self._train_input_transforms

    @property
    def test_input_transforms(self):
        if self._test_input_transforms is None:
            self._test_input_transforms = [
                Compose([
                    lambda x: x["image_arr"],
                    RepeatGrayscaleChannels(3),
                    Transpose(),
                    ToTensor(),
                    ToFloat(),
                ]),
            ]

        return self._test_input_transforms

    @property
    def train_label_transforms(self):
        if self._train_label_transforms is None:

            def to_numpy_array(x):
                index = x["skeletal_age"]
                if self.sex and x["male"]:
                    index += self.num_ages
                return np.array(index)

            self._train_label_transforms = (
                Compose([
                    to_numpy_array,
                    ToTensor(),
                    ToLong(),
                ])
            )

        return self._train_label_transforms

    @property
    def test_label_transforms(self):
        if self._test_label_transforms is None:

            def to_numpy_array(x):
                index = x["skeletal_age"]
                if self.sex and x["male"]:
                    index += self.num_ages
                return np.array(index)

            self._test_label_transforms = (
                Compose([
                    to_numpy_array,
                    ToTensor(),
                    ToLong(),
                ])
            )

        return self._test_label_transforms


    class Network(nn.Module):

        def __init__(self, num_classes, args):
            super().__init__()
            if args.network == "resnet":
                self._net = resnet50().cuda()
                self._net.fc = nn.Linear(2048, num_classes).cuda()
            if args.network == "densenet3":
                self._net = DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5).cuda()

        def forward(self, inputs):
            return self._net(inputs)

    class Criterion(nn.CrossEntropyLoss):

        pass


class DeVriesLarsonModelConfig(LarsonModelConfig):

    def __init__(self, args, hint_rate=0.0, lmbda=1.0):
        super().__init__(args)
        self.hint_rate = hint_rate
        self.lmbda = lmbda

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = self.Criterion(hint_rate=self.hint_rate,
                                             lmbda=self.lmbda)

        return self._criterion

    def task_metric(self, task_score_batch, metadata_batch):
        """
    output_batches consists of a batch of task scores and a batch of confidence
    scores
    """
        scores_batch = []
        for batch_iter, item in enumerate(metadata_batch["sex"], 0):
            if item == "F":
                scores = task_score_batch[batch_iter:batch_iter + 1, :self.num_ages]
            else:
                scores = task_score_batch[batch_iter:batch_iter + 1, self.num_ages:]

            scores_batch.append(torch.argmax(scores, dim=1))

        task_predictions = torch.cat(scores_batch).cpu().numpy()
        task_targets = metadata_batch["skeletal_age"].numpy()
        mad = abs(task_predictions - task_targets)
        return list(mad)


    class Network(nn.Module):

        def __init__(self, num_classes, args):
            super().__init__()
            if args.network == "resnet":
                self._net = resnet50().cuda()
                self._net.fc = nn.Linear(2048, num_classes).cuda()
            if args.network == "densenet3":
                self._net = DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5).cuda()

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

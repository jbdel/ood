import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torchvision.models.resnet import resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

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

  def __init__(self):
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
    self.reset_error_metrics()

  @property
  def optimizer(self):
    if self._optimizer is None:
      self._optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    return self._optimizer

  @property
  def scheduler(self):
    if self._scheduler is None:
      self._scheduler = ReduceLROnPlateau(self.optimizer,
                                          factor=0.1 ** 0.5,
                                          patience=16,
                                          verbose=True)

    return self._scheduler

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

  @property
  def error_metrics(self):
    raise NotImplementedError

  def update_error_metrics(self):
    raise NotImplementedError

  def reset_error_metrics(self):
    raise NotImplementedError


class LarsonModelConfig(ModelConfig):

  def __init__(self, age_range=(0, 240, 1), sex=True):
    """
    Inputs:
    - age_range: A Python 2-tuple or 3-tuple passed to range, that represents
      the predicted age classes.
    - sex: A Python bool that represents whether or not to separate predicted
      age classes by sex.
    """
    super().__init__()
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
      self._net = self.Network(self.num_classes)

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

  @property
  def error_metrics(self):
    return { "mad": np.mean(self._error_metrics["mad"]) }

  def update_error_metrics(self, output_batches, label_batch, metadata_batch):
    """
    In general, expects:
    - One batch per output of self.net.__call__
    - One batch of the label yielded by the dataset, according to the label
      transform
    - One batch of the metadata yielded by the dataset

    For this particular model, self.net.__call__ returns a single output
    consisting of pre-softmax scores for n categories, and self.label returns
    a numpy array with shape (1,) representing the desired index into n
    categories.
    """
    scores_batch = []
    for batch_iter, item in enumerate(metadata_batch["sex"], 0):
      if item == "F":
        scores = output_batches[batch_iter:batch_iter+1,:self.num_ages]
      else:
        scores = output_batches[batch_iter:batch_iter+1,self.num_ages:]

      scores_batch.append(scores.detach().cpu().numpy().squeeze())

    predictions = np.argmax(np.array(scores_batch), axis=1)
    targets = np.array(metadata_batch["skeletal_age"]).astype(int)

    error_metrics = { "mad": list(abs(predictions - targets)[metadata_batch["real"]]) }

    for k, v in self._error_metrics.items():
      v.extend(error_metrics[k])

  def reset_error_metrics(self):
    self._error_metrics = { "mad": [] }


  class Network(nn.Module):

    def __init__(self, num_classes):
      super().__init__()
      self._net = resnet50(pretrained=True).cuda()
      self._net.fc = nn.Linear(2048, num_classes).cuda()

    def forward(self, inputs):
      return self._net(inputs)

  class Criterion(nn.CrossEntropyLoss):

    pass


class DeVriesLarsonModelConfig(LarsonModelConfig):

  def __init__(self, hint_rate=0.0, lmbda=1.0):
    super().__init__()
    self.hint_rate = hint_rate
    self.lmbda = lmbda

  @property
  def criterion(self):
    if self._criterion is None:
      self._criterion = self.Criterion(hint_rate=self.hint_rate,
                                       lmbda=self.lmbda)

    return self._criterion

  @property
  def error_metrics(self):
    try:
      auroc = roc_auc_score(self._error_metrics["real_targets"],
                            self._error_metrics["confidence_probs"])
    except:
      auroc = None

    try:
      sorted_pairs = sorted(list(zip(self._error_metrics["real_targets"],
                                     self._error_metrics["confidence_probs"])),
                            key=lambda x: x[1])
      targets_sorted_by_probs = list(map(lambda x: x[0], sorted_pairs))
      count = np.ceil(0.05 * sum(targets_sorted_by_probs))
      for i in range(len(targets_sorted_by_probs)):
        if count <= 0:
          break

        if targets_sorted_by_probs[i]:
          count -= 1
      fn = sum(targets_sorted_by_probs[:i])
      tn = i - fn
      tp = sum(targets_sorted_by_probs[i:])
      fp = len(targets_sorted_by_probs) - i - tp
      fpr_at_tpr95 = fp/(fp+tn)
      detection_error = 0.5*(1-0.95) + 0.5*fp/(fp+tn)
    except:
      fpr_at_tpr95 = None
      detection_error = None

    try:
      fpr_1, tpr_1, _ = roc_curve(self._error_metrics["real_targets"],
                                  self._error_metrics["confidence_probs"],
                                  pos_label=1)
      fpr_0, tpr_0, _ = roc_curve(self._error_metrics["real_targets"],
                                  self._error_metrics["confidence_probs"],
                                  pos_label=0)
      auc_1 = auc(fpr_1, tpr_1)
      auc_0 = auc(fpr_0, tpr_0)
    except:
      auc_1 = None
      auc_0 = None

    return {
      "mad": np.mean(self._error_metrics["mad"]),
      "fpr_at_tpr95": fpr_at_tpr95,
      "detection_error": detection_error,
      "auroc": auroc,
      "aupr_in": auc_1,
      "aupr_out": auc_0,
    }

  def update_error_metrics(self, output_batches, label_batch, metadata_batch):
    """
    output_batches consists of a batch of task scores and a batch of confidence
    scores
    """
    task_score_batch, confidence_score_batch = output_batches

    scores_batch = []
    for batch_iter, item in enumerate(metadata_batch["sex"], 0):
      if item == "F":
        scores = task_score_batch[batch_iter:batch_iter+1,:self.num_ages]
      else:
        scores = task_score_batch[batch_iter:batch_iter+1,self.num_ages:]

      scores_batch.append(torch.argmax(scores, dim=1))

    task_predictions = torch.cat(scores_batch).cpu().numpy()
    task_targets = metadata_batch["skeletal_age"].numpy()

    confidence_probs = nn.Sigmoid()(confidence_score_batch).detach().cpu().numpy()
    real_targets = metadata_batch["real"].numpy()
    # only counts mad for real ones
    mad = abs(task_predictions - task_targets)[real_targets.astype(bool)]

    error_metrics = {
      "mad": list(mad),
      "confidence_probs": list(confidence_probs),
      "real_targets": list(real_targets)
    }

    for k, v in self._error_metrics.items():
      v.extend(error_metrics[k])

  def reset_error_metrics(self):
    self._error_metrics = {
      "mad": [],
      "confidence_probs": [],
      "real_targets": [],
    }


  class Network(nn.Module):

    def __init__(self, num_classes):
      super().__init__()
      self._net = resnet50(pretrained=True).cuda()
      self._net.fc = nn.Linear(2048, num_classes + 1).cuda()

    def forward(self, inputs):
      scores = self._net(inputs)
      confidence_score = scores[:,0]
      task_scores = scores[:,1:]

      return task_scores, confidence_score

  class Criterion(nn.Module):

    def __init__(self, hint_rate, lmbda):
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

    def forward(self, inputs, target):
      task_scores, confidence_score = inputs
      task_probs = nn.Softmax(dim=1)(task_scores)
      confidence_prob = nn.Sigmoid()(confidence_score)
      _, num_classes = task_scores.size()
      one_hot_target = nn.functional.one_hot(target, num_classes=num_classes)

      mask = (torch.rand_like(confidence_prob) < self.hint_rate).float()
      masked_confidence_prob = (mask * confidence_prob) + (1.0 - mask)
      masked_confidence_prob = masked_confidence_prob.unsqueeze(1)
      task_probs_pred = task_probs * masked_confidence_prob
      task_probs_hint = one_hot_target.float() * (1.0 - masked_confidence_prob)
      interpolated_task_probs = task_probs_pred + task_probs_hint

      task_loss = nn.NLLLoss()(torch.log(interpolated_task_probs), target)
      confidence_loss = -torch.log(masked_confidence_prob).mean()

      return task_loss + self.lmbda * confidence_loss

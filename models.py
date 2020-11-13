import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import *
from torch.optim.lr_scheduler import MultiStepLR
from densenet3 import DenseNet3
import torch.nn.functional as F


class DeVriesLarsonModelConfig(object):

    def __init__(self, args, hint_rate=0.0, lmbda=1.0, beta=0.3):
        super().__init__()
        self.args = args

        self.hint_rate = hint_rate
        self.beta = beta
        self.lmbda = lmbda

        self.criterion = self.Criterion(hint_rate=self.hint_rate,
                                        lmbda=self.lmbda,
                                        beta=self.beta,
                                        args=self.args)
        self.net = self.Network(args).cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.scheduler = None
        if self.args.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=0, factor=0.8)

    class Network(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.num_classes = args.num_classes
            self.dropout = nn.Dropout(p=0.5)
            if args.network == "resnet":
                self._net = resnet50(pretrained=False)
                in_features = self._net.fc.in_features
                self._net = torch.nn.Sequential(*list(self._net.children())[:-1])
                self.classifier = nn.Linear(in_features, self.num_classes)
                self.confidence = nn.Linear(in_features, 1)

        def forward(self, inputs):
            x = self._net(inputs)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            confidence_score = self.confidence(x)
            task_scores = self.classifier(x)

            return task_scores, confidence_score

    class Criterion(nn.Module):

        def __init__(self, hint_rate, lmbda, beta, args):
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
            self.beta = beta
            self.args = args
            self.prediction_criterion = nn.NLLLoss().cuda()

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

            if 'devries' in self.args.mode:
                if self.args.hint:
                    # b = torch.bernoulli(torch.empty(confidence_prob.size()).uniform_(0, 1)).cuda()
                    b = (torch.rand_like(confidence_prob) < self.args.hint_rate).float()
                    conf = confidence_prob * b + (1 - b)
                else:
                    conf = confidence_prob

                pred_new = task_probs * conf.expand_as(task_probs) + one_hot_target * (1 - conf.expand_as(one_hot_target))
                pred_new = torch.log(pred_new)
            else:
                pred_new = torch.log(task_probs)

            xentropy_loss = self.prediction_criterion(pred_new, target)

            if 'devries' in self.args.mode:
                confidence_loss = torch.mean(-torch.log(confidence_prob))
                total_loss = xentropy_loss + (self.lmbda * confidence_loss)
                if self.beta > confidence_loss.data:
                    self.lmbda = self.lmbda / 1.01
                elif self.beta <= confidence_loss.data:
                    self.lmbda = self.lmbda / 0.99
            else:
                confidence_loss = torch.tensor(0)
                total_loss = xentropy_loss

            return total_loss, xentropy_loss, confidence_loss

    class Criterion2(nn.Module):

        def __init__(self, hint_rate, lmbda, beta, args):
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
            self.beta = beta
            self.args = args
            self.prediction_criterion = nn.NLLLoss().cuda()

        def forward(self, inputs, target):
            task_scores, confidence_score = inputs
            task_probs = nn.Softmax(dim=1)(task_scores)
            confidence_prob = nn.Sigmoid()(confidence_score)
            _, num_classes = task_scores.size()

            eps = 1e-12
            task_probs = torch.clamp(task_probs, 0. + eps, 1. - eps)
            confidence_prob = torch.clamp(confidence_prob, 0. + eps, 1. - eps)

            one_hot_target = nn.functional.one_hot(target, num_classes=num_classes)

            mask = (torch.rand_like(confidence_prob) < self.hint_rate).float()
            masked_confidence_prob = (mask * confidence_prob) + (1.0 - mask)
            masked_confidence_prob = masked_confidence_prob.unsqueeze(1)
            task_probs_pred = task_probs * masked_confidence_prob
            task_probs_hint = one_hot_target.float() * (1.0 - masked_confidence_prob)
            interpolated_task_probs = task_probs_pred + task_probs_hint

            task_loss = nn.NLLLoss()(torch.log(interpolated_task_probs), target)
            confidence_loss = -torch.log(masked_confidence_prob).mean()

            total_loss = task_loss + self.lmbda * confidence_loss

            return total_loss, task_loss.cpu().data.numpy(), confidence_loss.cpu().data.numpy()

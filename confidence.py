import torch
import torch.nn as nn
from torch.autograd import Variable


def get_confidence(net, inputs, pred, confidence, args):

    if args.mode == 'baseline':
        probability = torch.softmax(pred, dim=-1)
        pred_value, _ = torch.max(probability.data, 1)
        out = pred_value.cpu().numpy()
    elif args.mode == 'devries':
        out = torch.sigmoid(confidence).data.cpu().numpy()
    elif args.mode == 'devries_odin':
        # https://arxiv.org/abs/1706.02690
        T = 1000.
        epsilon = 0.001

        images = Variable(inputs, requires_grad=True).cuda()
        images.retain_grad()

        net.zero_grad()
        pred, _ = net(images)
        _, pred_idx = torch.max(pred.data, 1)
        labels = Variable(pred_idx)
        pred = pred / T
        loss = nn.CrossEntropyLoss()(pred, labels)
        loss.backward()

        images = images - epsilon * torch.sign(images.grad)
        images = Variable(images.data, requires_grad=True)

        pred, _ = net(images)

        pred = pred / T
        pred = torch.softmax(pred, dim=-1)
        pred = torch.max(pred.data, 1)[0]
        out = pred.cpu().numpy()
    elif args.mode == 'oe' or args.mode == 'energy':
        out = (pred.mean(1) - torch.logsumexp(pred, dim=1)).data.cpu().numpy()
    else:
        raise NotImplementedError
    return out

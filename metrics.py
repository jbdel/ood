from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set_theme(style="darkgrid")


def auroc(preds, labels):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)
    return auc(fpr, tpr)


def aupr(preds, labels):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    precision, recall, _ = precision_recall_curve(labels, preds)
    return auc(recall, precision)


def num_95_tpr(preds, labels):
    fpr, tpr, _ = roc_curve(labels, preds)
    return sum(tpr > 0.95)


def fpr_at_95_tpr(preds, labels):
    """Return the FPR when TPR is at minimum 95%.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)


def detection_error(preds, labels):
    """Return the misclassification probability when TPR is 95%.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """
    fpr, tpr, _ = roc_curve(labels, preds)

    # Get ratio of true positives to false positives
    f2t_ratio = sum(np.array(labels) == 1) / len(labels)
    t2f_ratio = 1 - f2t_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    _detection_error = lambda idx: t2f_ratio * (1 - tpr[idx]) + f2t_ratio * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))


def calc_metrics(predictions, labels):
    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.

    These metrics conform to how results are reported in the paper 'Enhancing The
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.

        preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    """

    return {k: round(v, 3) for k, v in {
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels),
        'detection_error': detection_error(predictions, labels),
        'auroc': auroc(predictions, labels),
        'aupr_in': aupr(predictions, labels),
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels]),
        'num_95_tpr': num_95_tpr(predictions, labels),
    }.items()}


def plot_metrics(scores, labels, ind_confs, ood_confs, checkpoints_folder, name):
    os.makedirs(os.path.join(checkpoints_folder, 'roc'), exist_ok=True)
    os.makedirs(os.path.join(checkpoints_folder, 'pr'), exist_ok=True)
    # os.makedirs(os.path.join(checkpoints_folder, 'bar'), exist_ok=True)
    os.makedirs(os.path.join(checkpoints_folder, 'hist'), exist_ok=True)

    plot_roc(scores, labels).savefig(os.path.join(checkpoints_folder, 'roc', name))
    plot_pr(scores, labels).savefig(os.path.join(checkpoints_folder, 'pr', name))
    # plot_barcode(scores, labels).savefig(os.path.join(checkpoints_folder, 'bar', name))
    plot_devries_histograms(ind_confs, ood_confs, scores).savefig(os.path.join(checkpoints_folder, 'hist', name))
    plt.close('all')

def plot_roc(preds, labels, title="Receiver operating characteristic"):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return plt


def plot_pr(preds, labels, title="Precision recall curve"):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    return plt


def plot_classification(corr, conf, checkpoints_folder, name, bins=50):
    # Plot histogram of correctly classified and misclassified examples
    os.makedirs(os.path.join(checkpoints_folder, 'classif'), exist_ok=True)
    plt.figure()
    sns.histplot(conf[corr].ravel(), kde=False, bins=bins, stat="density", label='Correct', color="#AECFBA")
    sns.histplot(conf[np.invert(corr)].ravel(), kde=False, bins=bins, stat="density", label='Incorrect', color="#AABAD7")
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(checkpoints_folder, 'classif', name))


def plot_barcode(preds, labels):
    """Plot a visualization showing inliers and outliers sorted by their prediction of novelty."""
    # the bar
    x = sorted([a for a in zip(preds, labels)], key=lambda x: x[0])
    x = np.array([[49, 163, 84] if a[1] == 1 else [173, 221, 142] for a in x])
    # x = np.array([a[1] for a in x]) # for bw image

    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary_r, interpolation='nearest')

    fig = plt.figure()

    # a horizontal barcode
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x.reshape((1, -1, 3)), **barprops)

    return fig


def plot_devries_histograms(ind_scores, ood_scores, scores, name='histogram'):
    # Plot histogram of correctly classified and misclassified examples in visdom
    ranges = (np.min(scores), np.max(scores))
    plt.figure()
    sns.histplot(ind_scores.ravel(), binrange=ranges, kde=False, bins=50, stat="density",
                 label='In-distribution', color="#AECFBA")

    sns.histplot(ood_scores.ravel(), binrange=ranges, kde=False, bins=50, stat="density",
                 label='Out-of-distribution', color="#AABAD7")

    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title(name)
    plt.legend()

    return plt

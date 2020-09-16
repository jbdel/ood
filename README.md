# Using Out of Distribution Detection to Fix Nearly All AI Models in Medical Imaging

The code is based on the following paper:

```
@article{devries2018learning,
  title={Learning confidence for out-of-distribution detection in neural networks},
  author={DeVries, Terrance and Taylor, Graham W},
  journal={arXiv preprint arXiv:1802.04865},
  year={2018}
}
```

<img src="https://i.imgur.com/obcU1Ez.jpg" width="50%"/>

## Dataset 
Download the preprocessed data [[here]](https://www.dropbox.com/s/uvsuklukg3iu513/data.zip?dl=1)(0.5 Go) from the [RSNA Bone Age](https://www.kaggle.com/kmader/rsna-bone-age) and the [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/).

Unzip the file and place it in the data folder: ```unzip data.zip -d data```<br/>
Each example in the `train.csv` and `test.csv` corresponds to an image in `.npz` format (preprocessed as `uint8` of size `256x256`). 

## Experiments
Two experiments can be carried out.

### Detecting out-of-distribution images for pediatric hand radiographs

Train samples are from RNSA hand radiograph and we use as test set the concatenation of the RNSA test and MURA train set.

```
python train.py --experiment_name=random_transforms__hint_rate_0.5__lmbda_0.05 --num_epochs=50 --hint_rate=0.5 --lmbda=0.05 --model devries
```


Best epoch | MAD | fpr_at_tpr95 | detection_error | auroc | aupr_in | aupr_out
------------ | ------------- | ------------- | ------------- | -------------| -------------| -------------
16  | 25.23  | 0.95  | 0.5 | 0.7922 | 0.7922 | 0.207

```diff
- Experiment nok
```

### Metrics

1. **FPR at 95% TPR** is the probability that a negative (out-of-distribution) example is misclassified as positive (in-distribution) when the true positive rate (TPR) is as high as 95%. True positive rate can be computed by TPR = TP / (TP+FN), where TP and FN denote true positives and false negatives respectively. The false positive rate (FPR) can be computed by FPR = FP / (FP+TN), where FP and TN denote false positives and true negatives respectively.

2. **Detection Error** is the misclassification probability when TPR is 95%, given by 0.5(1-TPR) + 0.5FPR, where positive and negative examples have equal probability of appearing in the test set.

3. **AUROC** is the Area Under the Receiver Operating Characteristic curve, which is also a threshold-independent metric. The ROC curve depicts the relationship between TPR and FPR. The AUROC can be interpreted as the probability that a positive example is assigned a higher detection score than a negative example. A perfect detector corresponds to an AUROC score of 100%.

4. **AUPR** is the Area under the Precision-Recall curve, which is another threshold independent metric. The PR curve is a graph showing the precision=TP/(TP+FP) and recall=TP/(TP+FN) against each other. The metric AUPR-In and AUPR-Out denote the area under the precision-recall curve where in-distribution and out-of-distribution images are specified as positives, respectively


### Detecting out-of-distribution images for AP/PA view chest radiographs

Running...

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
Download the preprocessed data [[here]] from the [RSNA Bone Age](https://www.kaggle.com/kmader/rsna-bone-age) and the [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/).

Unzip the file and place it in the data folder: ```unzip data.zip -d data```<br/>
Each example in the `train.csv` and `test.csv` corresponds to an image in `.npz` format (preprocessed as `uint8` of size `256x256`. 

## Experiments
Two experiments can be carried out

### Detecting out-of-distribution images for pediatric hand radiographs

Train samples are from RNSA hand radiograph and we use as test set the concatenation of the RNSA test and MURA train set.

```
python train.py --experiment_name=random_transforms__hint_rate_0.5__lmbda_0.05 --num_epochs=50 --hint_rate=0.5 --lmbda=0.05 --model devrie
```

Epoch | MAD | fpr_at_tpr95 | detection_error | auroc | aupr_in | aupr_out
------------ | ------------- | ------------- | ------------- | -------------| -------------| -------------
x | x  | x  | x  | x  | x  | x 
```diff
+ Experiment ok
```

### Detecting out-of-distribution images for AP/PA view chest radiographs

Running...

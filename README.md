# pSTN-baselines

This is the official Pytorch implementation for the Probabilistic Spatial Transformer Network https://arxiv.org/abs/2004.03637

### Abstract

Spatial Transformer Networks (STNs) estimate image transformations that can improve downstream tasks by `zooming in' on relevant regions in an image. However, STNs are hard to train and sensitive to mis-predictions of transformations. To circumvent these limitations, we propose a probabilistic extension that estimates a stochastic transformation rather than a deterministic one. Marginalizing transformations allows us to consider each image at multiple poses, which makes the localization task easier and the training more robust. As an additional benefit, the stochastic transformations act as a localized, learned data augmentation that improves the downstream tasks. We show across standard imaging benchmarks and on a challenging real-world dataset that these two properties lead to improved classification performance, robustness and model calibration. We further demonstrate that the approach generalizes to non-visual domains by improving model performance on time-series data.

# Train & Test

All experiments in the paper can be found in the script folder
```
cd pSTN-baselines;
bash script/[name-of-experiment].sh 
```
for example 

```
cd pSTN-baselines;
bash script/UAI_fashion_mnist_pretrain.sh
```

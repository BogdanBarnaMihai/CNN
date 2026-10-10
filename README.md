# CNN — A Deep Dive into Convolutional Neural Networks

An empirical study of CNNs across four datasets and eight experiments, covering classification, regression, translation invariance, feature-map interpretability, custom loss functions, and convolutional autoencoders.

Built with PyTorch and inspired by the teaching style of Mike X Cohen's *A deep understanding of deep learning*.

---

## Overview

This project is an attempt to understand what CNNs actually learn, not just how to train them. Each experiment isolates one specific idea — translation invariance, kernel size, channel width, feature diversity, denoising — and tries to quantify the effect with a concrete number.

| # | Experiment | Dataset | Key Result |
|:-:|:-----------|:--------|:-----------|
| 1 | MNIST baseline | MNIST | ~98% test accuracy |
| 2 | Translation invariance (CNN vs MLP) | Shifted MNIST | CNN about 20 points higher than MLP |
| 3 | Gaussian regression | Synthetic Gaussians | Pearson r = 0.98 |
| 4 | Hyperparameter sweep | Synthetic Gaussians | Diminishing returns past ~200 units |
| 5 | Feature-map visualization | Synthetic Gaussians | Conv2 filters roughly 40% correlated |
| 6 | Custom loss functions | Synthetic Gaussians | MSE best for reconstruction |
| 7 | Occlusion removal | Occluded Gaussians | About 60% MSE reduction |
| 8 | EMNIST letters | EMNIST | 90%+ accuracy across 26 classes |

## Experiments

### 1. MNIST Baseline
A two-conv-layer CNN (`Conv -> Pool -> Conv -> Pool -> FC -> FC`) reaching roughly 98% test accuracy in 10 epochs. Nothing fancy, just a solid reference point.

### 2. Translation Invariance — CNN vs MLP
Compare a CNN against an equivalent fully-connected network on horizontally shifted digits. The CNN keeps its accuracy far better. This is the classic evidence of the built-in translation invariance that comes from convolution plus pooling.

### 3. Gaussian Regression
Regress the center coordinates (Cx, Cy) and the width of a 2D Gaussian blob. A `FlexibleCNN` with three output units gets r > 0.98 on all three parameters, which was honestly a bit surprising given how much noise is in the images.

### 4. Hyperparameter Sweep
A grid search over conv-channel configurations and fully-connected widths. Accuracy improves as you add channels, up to a point, then flattens out. Good reminder that more parameters is not automatically better.

### 5. Feature-Map Visualization
Extract and visualize the intermediate activations from both convolutional layers. Then compute inter-filter correlation matrices to see how much redundancy exists in the deeper layers.

### 6. Custom Loss Functions
Benchmark MSE, L1, an L2-plus-average penalty, and a negative-correlation loss on the same convolutional autoencoder. A small reminder that the loss function really does shape the output statistics.

### 7. Occlusion Removal
Train a convolutional autoencoder to remove randomly placed bars from Gaussian images. The network learns to reconstruct the clean image from a corrupted input, though it does not magically generalize to occlusion patterns it has never seen.

### 8. EMNIST Letters
A three-conv-layer CNN with BatchNorm and Dropout classifies 26 letters. The per-class confusion matrix shows which letters are hardest to tell apart — as expected, pairs like *i* and *l* dominate the mistakes.

---

## Tech Stack

- PyTorch for model building and training
- NumPy for numerical operations
- scikit-learn for train/test splits and metrics
- matplotlib for visualization
- torchsummary for model summaries
- torchvision for the EMNIST dataset

## Results at a Glance

| Metric | Value |
|:---|:---|
| MNIST test accuracy | ~98% |
| EMNIST test accuracy | 90%+ |
| Gaussian regression r | 0.98 |
| CNN vs MLP on shifted MNIST | about +20 points |
| Occlusion-removal MSE reduction | around 60% |

---

## What I Learned

- CNNs are inherently translation-invariant. Pooling plus convolution gives robustness to shifts that MLPs simply lack.
- Channel width matters more than depth on small datasets, up to a point. After that, the returns flatten quickly.
- Deeper feature maps become correlated with each other. That is a sign of representational redundancy that could probably be pruned.
- The loss function shapes output statistics. L1 tends to produce sparse outputs, MSE produces smoother ones, and correlation loss aligns variances.
- Autoencoders are good denoisers, but they do not magically generalize to occlusion patterns they were not trained on.

---

## Future Work

- Add data augmentation to the MNIST pipeline: rotation, shear, elastic distortion
- Replace the classifier head with a small transformer and compare
- Explore Grad-CAM for class-activation visualization
- Extend Experiment 7 to a variational autoencoder (VAE) for probabilistic denoising

---

## References

- LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (1998)
- Mike X Cohen, *A deep understanding of deep learning* (Udemy course)
- Cohen et al., *EMNIST: an extension of MNIST to handwritten letters* (2017)

---

## License

MIT. Free to use, modify, and share.

---

## Acknowledgments

Thanks to Mike X Cohen for the pedagogical approach that inspired this project, and to the open-source PyTorch community.

---

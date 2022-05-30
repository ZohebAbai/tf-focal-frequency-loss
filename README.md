# Focal Frequency Loss - Tensorflow Implementation

![teaser](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/resources/teaser.jpg)

This repository provides the Tensorflow implementation for the following paper:

**Focal Frequency Loss for Image Reconstruction and Synthesis** by Liming Jiang, Bo Dai, Wayne Wu and Chen Change Loy 
in ICCV 2021.

[**Project Page**](https://www.mmlab-ntu.com/project/ffl/index.html) | [**Paper**](https://arxiv.org/abs/2012.12821) | [**Poster**](https://liming-jiang.com/projects/FFL/resources/poster.pdf) | [**Slides**](https://liming-jiang.com/projects/FFL/resources/slides.pdf) | [**YouTube Demo**](https://www.youtube.com/watch?v=RNTnDtKvcpc) | [**Official PyTorch Implementation**](https://github.com/EndlessSora/focal-frequency-loss)

> **Abstract:** *Image reconstruction and synthesis have witnessed remarkable progress thanks to the development of generative models. Nonetheless, gaps could still exist between the real and generated images, especially in the frequency domain. In this study, we show that narrowing gaps in the frequency domain can ameliorate image reconstruction and synthesis quality further. We propose a novel focal frequency loss, which allows a model to adaptively focus on frequency components that are hard to synthesize by down-weighting the easy ones. This objective function is complementary to existing spatial losses, offering great impedance against the loss of important frequency information due to the inherent bias of neural networks. We demonstrate the versatility and effectiveness of focal frequency loss to improve popular models, such as VAE, pix2pix, and SPADE, in both perceptual quality and quantitative performance. We further show its potential on StyleGAN2.*


## Quick Start

Run `pip install tf-focal-frequency-loss` for installation. Then, the following code is all you need.

```python
import tensorflow as tf
from tf_focal_frequency_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize tf.keras.layers.Layer class

fake = tf.random.normal((4, 3, 64, 64))  # replace it with the predicted tensor of shape (N, C, H, W)
real = tf.random.normal((4, 3, 64, 64))  # replace it with the target tensor of shape (N, C, H, W)

loss = ffl(fake, real)  # calculate focal frequency loss
```


## License

All rights reserved. The code is released under the [MIT License](https://github.com/ZohebAbai/tf-focal-frequency-loss/blob/main/LICENSE).

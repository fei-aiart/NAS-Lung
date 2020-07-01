# NAS-Lung
> 3D NAS for Pulmonary Nodules Classification
>
> Jiang et al. Learning Efficient, Explainable and Discriminative Representations for Pulmonary Nodules Classification. (under review)

## Abstract: 

Automatic pulmonary nodules classification is significant for early diagnosis of lung cancers. Recently, deep learning techniques have enabled remarkable progress in this field. However, these deep models are typically of high computational complexity and work in a black-box manner. To combat these challenges, in this work, we aims to build an efficient and (partially) explainable classification model. Specially, we first use neural architecture search (NAS) to automatically search 3D network architectures with excellent accuracy/speed trade-off. Afterwards, we expand the convolutional block attention module (CBAM) to the searched networks, which helps us understand the reasoning process. During training, we use A-Softmax loss to learn angularly discriminative representations. In the inference stage, we employ an ensemble of diverse neural networks to improve the prediction accuracy and robustness. We conduct extensive experiments on the LIDC-IDRI database. Compared with previous state-of-the-art, our model shows superior performance in general by using less than 1/10 parameters. Besides, empirical study shows that the reasoning process of learned networks is in conformity with physicians' diagnosis. Related code and results have been released at: https://github.com/fei-hdu/NAS-Lung.

## Requirements

- Pytorch


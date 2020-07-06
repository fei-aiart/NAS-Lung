# NAS-Lung
> 3D NAS for Pulmonary Nodules Classification
>
> Jiang et al. Learning Efficient, Explainable and Discriminative Representations for Pulmonary Nodules Classification. (under review)

## Architecture

![Architecture](imgs/architecture.png)

## Results

### NASLung9

| model               | Accu. | Sens. | Spec. | F1 Score | para.(M) |
| ------------------- | ----- | ----- | ----- | -------- | -------- |
| Multi-crop CNN      | 87.14 | -     | -     | -        | -        |
| Nodule-level 2D CNN | 87.30 | 88.50 | 86.00 | 87.23    | -        |
| Vanilla 3D CNN      | 87.40 | 89.40 | 85.20 | 87.25    | -        |
| DeepLung            | 90.44 | 81.42 | -     | -        | 141.57   |
| AE-DPN              | 90.24 | 92.04 | 88.94 | 90.45    | 678.69   |
|                     |       |       |       |          |          |
| 3D-NAS (ours)       | 88.71 | 85.85 | 91.17 | 87.20    | 7.99     |
| NASLung9 (ours)     | 91.72 | 90.21 | 92.85 | 90.81    | 63.53    |

### 3D-NAS

| Model   | Accu. | Sens. | Spec. | F1 Score | para. |
| ------- | ----- | ----- | ----- | -------- | ----- |
| Model-A | 88.71 | 85.85 | 91.17 | 87.20    | 7.99  |
| Model-B | 88.16 | 87.11 | 88.61 | 86.90    | 8.05  |
| Model-C | 88.46 | 83.83 | 92.15 | 86.68    | 8.05  |
| Model-D | 87.59 | 83.64 | 90.52 | 85.85    | 0.63  |
| Model-E | 88.27 | 82.41 | 92.92 | 86.17    | 7.79  |
| Model-F | 88.13 | 87.86 | 88.89 | 86.88    | 11.28 |
| Model-G | 88.61 | 85.41 | 91.40 | 87.02    | 11.33 |
| Model-H | 87.94 | 83.79 | 91.10 | 86.13    | 11.28 |
| Model-I | 88.11 | 86.08 | 89.73 | 86.64    | 11.33 |

## Prerequisites

* Linux or similar environment
* Python 3.7
* Pytorch 0.4.1
* NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

* Clone this repo:
    ```shell script
    git clone https://github.com/fei-hdu/NAS-Lung
    cd NAS-Lung
    ```
* Install PyTorch 0.4+ and torchvision from http://pytorch.org and other dependencies (e.g., visdom and dominate). You can install all the dependencies by

    ```shell script
    pip install -r requirments.txt
    ```
* Download Dataset [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

### Neural Architecture Search

```shell script
python search_main.py --train_data_path {train_data_path}  --test_data_path {test_data_path} --save_module_path {save_module_path}
```

### Train/Test

* Train a model
    ```shell script
    sh run_training.sh
    ```

* Test a model
    ```shell script
    python test.py --test_data_path {test_data_path} --preprocess_path {preprocess_path} --model_path {model_path}
    ```

### DataSet

* [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

### Model Result

- our final result can be download:[Google Drive](https://drive.google.com/drive/folders/1vUFi5tEfMcDcKqMbxuN3Tt44QwLcDZnA?usp=sharing)

### Training/Test Tips

- Best practice for training and testing your models. 
- Feel free to ask any questions about ***coding***. **Fuhao Shen, ``1048532267sfh@gmail.com``**

## Acknowledgement

- Our work/code is inspired by [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search, CVPR 2019](https://github.com/lixincn2015/Partial-Order-Pruning).

## Selected References

- S. Armato III, G. et al., Data from **LIDC-IDRI**, The Cancer Imaging Archivedoi:http://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX. URL https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI.
- X. Li, Y. Zhou, Z. Pan, J. Feng, **Partial order pruning**: For best speed/accuracy trade-off in neural architecture search (2019) 9145–9153.
- S. Woo, J. Park, J.-Y. Lee, I. So Kweon, **CBAM**: Convolutional block attention module, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 3–19.
- W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, L. Song, **Sphereface**: Deep hypersphere embedding for face recognition, in: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- T. Elsken, J. H. Metzen, F. Hutter, **Neural architecture search**: A survey, Journal of Machine Learning Research 20 (55) (2019) 1–21.
- W. Zhu, C. Liu, W. Fan, X. Xie, **Deeplung**: Deep 3d dual path nets for automated pulmonary nodule detection and classification, in: 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE, 2018, pp. 673–681.
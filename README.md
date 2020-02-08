# Any-Precision Deep Neural Network
Official code and models in PyTorch for the paper [Any-Precision Deep Neural Networks](https://arxiv.org/abs/1911.07346).

## Run
#### Environment
* Python 3.7
* PyTorch 1.1.0
* torchvision 0.2.1
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [gpustat](https://github.com/wookayin/gpustat)

#### Train
##### Resnet-20 Models on CIFAR10
Run the script below and dataset will download automatically.
 ```
 ./train_cifar10.sh
 ```

##### SVHN Models on SVHN
Run the script below and dataset will download automatically.
 ```
 ./train_svhn.sh
 ```

##### Resnet-50 Models on ImageNet
Before running the script below, one needs to manually download ImageNet and save it properly according to `data_paths` in `dataset/data.py`.
 ```
 ./train_imagenet.sh
 ```

#### Test
To test a trained model, simply run the corresponding training script for one epoch with pretrained model loaded and without the training part.

## Trained Models
Due to the following listed training hyperparameter changes, numbers below may be different from those in the paper.
* Init lr for any-precision models: 0.1 -> 0.5.
* We use ReLU for 32-bit model instead of Clamp (check [here](https://github.com/haichaoyu/any-precision-nets/blob/master/models/resnet_quan.py#L22)).
* We use tanh nonlinearity for 32-bit model for consistency with other precisions (check [here](https://github.com/haichaoyu/any-precision-nets/blob/master/models/quan_ops.py#L88)).

##### Resnet-20 Models on CIFAR10
| Models                               | 1 bit | 2 bit | 4 bit | 8 bit | FP32  |
|--------------------------------------|-------|-------|-------|-------|-------|
| Resnet-20                            | [91.50](https://www.dropbox.com/s/81efiwknrxjfz9j/resnet20q_1.pth.tar?dl=0) | [93.26](https://www.dropbox.com/s/x7u2cpye4zp3u7r/resnet20q_2.pth.tar?dl=0) | [93.62](https://www.dropbox.com/s/h8o7c9dykq82m9u/resnet20q_4.pth.tar?dl=0) | [93.42](https://www.dropbox.com/s/5k3e4sztgo9tvko/resnet20q_8.pth.tar?dl=0) | [93.58](https://www.dropbox.com/s/2q0buwb36eqfup5/resnet20q_32.pth.tar?dl=0) |
| [Resnet-20-Any (hard<sup>1</sup>)](https://www.dropbox.com/s/08jcbc43e5kgl4w/resnet20q_any_hard.pth.tar?dl=0)     | 91.48 | 93.74 | 93.87 | 93.92 | 93.71 |
| [Resnet-20-Any (soft<sup>2</sup>)](https://www.dropbox.com/s/xg2xw5tburppftf/resnet20q_any_soft.pth.tar?dl=0)     | 91.18 | 93.51 | 93.21 | 93.13 | 93.63 |
| [Resnet-20-Any (recursive<sup>3</sup>)](https://www.dropbox.com/s/jx93fr74wfwtxs6/resnet20q_any_recursive.pth.tar?dl=0)| 91.89 | 93.90 | 93.86 | 93.75 | 94.11 |

1: Softmax Cross Entropy Loss  
2: Softmax Cross Entropy Loss with FP32 prediction as supervision  
3: Softmax Cross Entropy Loss with higher-precision model as supervision for   lower-precision model  

##### SVHN Models on SVHN
| Models                   | 1 bit | 2 bit | 4 bit | 8 bit | FP32  |
|--------------------------|-------|-------|-------|-------|-------|
| SVHN                     | [90.94](https://www.dropbox.com/s/qji6bav9wbdduav/svhnq_1.pth.tar?dl=0) | [96.45](https://www.dropbox.com/s/6nficennhvi988y/svhnq_2.pth.tar?dl=0) | [97.04](https://www.dropbox.com/s/3qggq03fn0z89lb/svhnq_4.pth.tar?dl=0) | [97.04](https://www.dropbox.com/s/vxyxwuf29ro011u/svhnq_8.pth.tar?dl=0) | [97.10](https://www.dropbox.com/s/eaex2jrhywhx61r/svhnq_32.pth.tar?dl=0) |
| [SVHN-Any (hard)](https://www.dropbox.com/s/0exa8t7y0a9c4zg/svhnq_any_hard.pth.tar?dl=0)          | 88.98 | 95.54 | 96.71 | 96.72 | 96.60 |
| [SVHN-Any (soft)](https://www.dropbox.com/s/yfnpuc3t86iw3nr/svhnq_any_soft.pth.tar?dl=0)          | 88.49 | 94.62 | 96.13 | 96.20 | 96.17 |
| [SVHN-Any (recursive)](https://www.dropbox.com/s/x9mxt69s2bmlzuz/svhnq_any_recursive.pth.tar?dl=0)     | 88.21 | 94.94 | 96.19 | 96.22 | 96.29 |

##### Resnet-50 Models on ImageNet
| Models                   | 1 bit | 2 bit | 4 bit | 8 bit | FP32  |
|--------------------------|-------|-------|-------|-------|-------|
| Resnet-50                |[57.83](https://www.dropbox.com/s/qpywwky7cx3jtsk/resnet50q_1.pth.tar?dl=0)<sup>4</sup>|[68.74](https://www.dropbox.com/s/93uthplop94box2/resnet50q_2.pth.tar?dl=0)<sup>4</sup>|[74.12](https://www.dropbox.com/s/j43l2zyxkj7auqa/resnet50q_4.pth.tar?dl=0)<sup>5</sup>|[74.96](https://www.dropbox.com/s/q879ap3auofcgo2/resnet50q_8.pth.tar?dl=0)<sup>5</sup>|[75.95](https://www.dropbox.com/s/t74fbzsxxs0bkk4/resnet50q_32.pth.tar?dl=0)<sup>5</sup>|
| [Resnet-50-Any (recursive)](https://www.dropbox.com/s/df87f40od0g8uq9/resnet50q_any_recursive.pth.tar?dl=0)|58.77            |71.66            |73.84            |74.07            |74.63 |

4: Weight decay 1e-5  
5: Weight decay 1e-4

## Citation
If you find this repository helpful, please consider citing our paper:
```
@article{yu2019any,
  title={Any-Precision Deep Neural Networks},
  author={Yu, Haichao and Li, Haoxiang and Shi, Honghui and Huang, Thomas S and Hua, Gang},
  journal={arXiv preprint arXiv:1911.07346},
  year={2019}
}
```

## Contact
Please feel free to contact Haichao Yu at haichaoyu3@gmail.com for any issue.

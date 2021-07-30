## ESNet: An  Efficient  Stereo  Matching  Network

Code in PyTorch for paper "ES-Net:  An  Efficient  Stereo  Matching  Network" submitted to IROS 2021 [[Paper Link]](https://arxiv.org/abs/2103.03922). 

## Dependency
Python 3.7

PyTorch(1.3.0+)

torchvision 0.2.0

## Installation

1. Ensure you have cuda/10.2 and gcc 7.5.
2. Go to networks/deform_conv and run 'python3 setup.py install --user' to build the egg package.
3. Then install the package by running 'python3 -m pip install --users .'
4. Now go to layers_package and run 'bash install.sh'.
5. Next go to both channelnorm_package and resample2d_package subdirs and run 'python3 -m pip install --users .'
6. Now come back to root directory of the project and run 'bash compile.sh'.

## Important links
1. networks/deform_conv: https://github.com/CHONSPQX/modulated-deform-conv
2. deform_conv_cuda fix for AT_CHECK: https://github.com/mrlooi/rotated_maskrcnn/issues/31
3. Flownet repo (layers_package): https://github.com/NVIDIA/flownet2-pytorch
4. Flownet custom kernel gcc fix: https://discuss.pytorch.org/t/error-when-building-custom-cuda-kernels-with-pytorch-1-6-0/94120/3

## Usage
"networks/ESNet.py" and "networks/ESNet_M.py" contains the implementation of the proposed efficient stereo matching network.

To train the ESNet on SceneFlow dataset, you will need to modify the path to the dataset in the "exp_configs/esnet_sceneflow.conf". Then run
```
dnn=esnet_sceneflow bash train.sh
```

## Citation
If you find the code and paper is useful in your work, please cite
```
@misc{huang2021esnet,
    title={ES-Net: An Efficient Stereo Matching Network},
    author={Zhengyu Huang and Theodore B. Norris and Panqu Wang},
    year={2021},
    eprint={2103.03922},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

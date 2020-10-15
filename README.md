
# PFLD - Ultralight - 275FPS

------

## 实现功能
* PFLD的训练/测试/评估/ncnn C++推理
* PFLD_Ulitralight的训练/测试/评估/ncnn C++推理
* 人脸98个关键点检测
* 支持onnx导出
* 网络parameter和flop计算

## 基于PFLD进行优化的超轻量级人脸关键点检测器
提供了一系列适合移动端部署的人脸关键点检测器： 对PFLD网络的基础结构进行优化，并调整网络结构，使其更适合边缘计算，而且在绝大部分情况下精度均好于原始PFLD网络。

可见PFLD基础网络中使用的是MobileNet V2的Inverted Residual Block，但是整体网络结构与MobileNet V2相比会有所差异，PFLD会比MobileNet V2更浅更窄，因此理论上PFLD比MobileNet V2更快速。

## 运行环境
* Ubuntu 16.04
* Python 3.7
* Pytorch 1.4
* CUDA 10.1

如果想省去环境搭建工作，可以使用以下Docker环境: [Docker_PFLD](https://hub.docker.com/r/tankrant/pfld_pytorch/tags)


## 精度
WFLW测试结果
|Model|Width|NME|Inference Time(NCNN 1xCore)|Model Size|
|:----:|:----:|:----:|:----:|:----:|
|PFLD|0.25<br>1|0.06163<br>0.05818|5.74ms<br>39.3ms|0.45M<br>5M|
|PFLD_Ghost|0.25<br>1|0.06072 (&darr;1.48%)<br>0.05652 (&darr;2.85%)|3.63ms (&darr;36.8%)<br>18.1ms   (&darr;53.9%)|0.41M (&darr;8.89%)<br>3.40M (&darr;32.0%)|

## Installation
**Clone and install:**
* git clone https://github.com/AnthonyF333/PFLD_UltraLight.git
* Pytorch version 1.4.0 are needed.
* Codes are based on Python 3.7

**Data:**
* Download WFLW dataset: 
  * [WFLW.zip](https://pan.baidu.com/s/1WHSwQOqbf9QQWcoLgEQbng) 
  * Password: rw1t
* Move the WFLW.zip to ./data/ directory and unzip the WFLW.zip
* Run SetPreparation.py. 
　　
* By default, it repeat 10 times for every image for augmentation, and save result in ./test_data/ and ./train_data/ directory.
　　
## Training
Before training, you can check or modify network configuration (e.g. batch_size, epoch and steps etc..) in config.py.
  * MODEL_TYPE: you can choose PFLD, PFLD_Ghost or PFLD_Ghost_Slim for different network.
  * WIDTH_FACTOR: you can choose 1 for original network or 0.25 for narrower network.

After modify the configuration, run train.py to start training.

## Test
Before test, modify the configuration in test.py, include the model_path, test_dataset, model_type, width_factor etc.
Then run test.py to align the images in the test dataset, and save result in current directory.

## C++ inference ncnn
Generate the onnx file: Modify the model_type, width_factor, model_path in pytorch2onnx.py, and then run pytorch2onnx.py.

## References
* [PFLD](https://github.com/polarisZhao/PFLD-pytorch)
* [GhostNet](https://github.com/huawei-noah/ghostnet)

------


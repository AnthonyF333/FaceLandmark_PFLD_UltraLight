
# PFLD - Ultralight - 400FPS(CPU)

------

!!2022年结合MobileOne的性能更加优异的人脸关键点检测算法已开源->[PFLD_GhostOne](https://github.com/AnthonyF333/PFLD_GhostOne)

## 实现功能
* PFLD的训练/测试/评估/ncnn C++推理
* PFLD_Ulitralight的训练/测试/评估/ncnn C++推理
* 人脸98个关键点检测
* 支持onnx导出

![image](https://github.com/AnthonyF333/PFLD_UltraLight/blob/master/images/tim_align.gif)

## 基于PFLD进行优化的超轻量级人脸关键点检测器
提供了一系列适合移动端部署的人脸关键点检测器： 对PFLD网络的基础结构进行优化，并调整网络结构，使其更适合边缘计算，而且在绝大部分情况下精度均好于原始PFLD网络，在CPU下运行最快可达到400fps。

## 运行环境
* Ubuntu 16.04
* Python 3.7
* Pytorch 1.4
* CUDA 10.1

如果想省去环境搭建工作，可以使用以下Docker环境: [Docker_PFLD](https://hub.docker.com/r/tankrant/pfld_pytorch/tags)


## 精度
WFLW测试结果

输入大小为112x112
|Model|Width|NME|Inference Time(NCNN 1xCore)|Model Size|
|:----:|:----:|:----:|:----:|:----:|
|PFLD|0.25<br>1|0.06159<br>0.05837|5.5ms<br>39.7ms|0.45M<br>5M|
|PFLD_Ultralight|0.25<br>1|0.06101 (&darr;0.94%)<br>0.05749 (&darr;1.51%)|3.6ms (&darr;34.5%)<br>18.6ms   (&darr;53.1%)|0.41M (&darr;8.89%)<br>3.40M (&darr;32.0%)|
|PFLD_Ultralight_Slim|0.25<br>1|0.06258 (&uarr;1.61%)<br>0.05627 (&darr;3.60%)|3.3ms (&darr;40.0%)<br>16.0ms   (&darr;59.7%)|0.38M (&darr;15.6%)<br>3.10M (&darr;38.0%)|


输入大小为96x96
|Model|Width|NME|Inference Time(NCNN 1xCore)|Model Size|
|:----:|:----:|:----:|:----:|:----:|
|PFLD|0.25<br>1|0.06136<br>0.05818|4.1ms<br>29.2ms|0.43M<br>4.8M|
|PFLD_Ultralight|0.25<br>1|0.06321 (&uarr;3.0%)<br>0.05745 (&darr;1.25%)|2.7ms (&darr;34.1%)<br>14.0ms   (&darr;52.1%)|0.39M (&darr;9.30%)<br>3.20M (&darr;33.3%)|
|PFLD_Ultralight_Slim|0.25<br>1|0.06402 (&uarr;4.33%)<br>0.05683 (&darr;2.32%)|2.5ms (&darr;40.0%)<br>12.0ms   (&darr;59.7%)|0.37M (&darr;14.0%)<br>2.90M (&darr;39.6%)|

## Installation
**Clone and install:**
* git clone https://github.com/AnthonyF333/PFLD_UltraLight.git
* cd ./PFLD_UltraLight
* pip install -r requirement.txt
* Pytorch version 1.4.0 are needed.
* Codes are based on Python 3.7

**Data:**
* Download WFLW dataset: 
  * [WFLW.zip](https://drive.google.com/file/d/1XOcAi1bfYl2LUym0txl_A4oIRXA_2Pf1/view?usp=sharing)
* Move the WFLW.zip to ./data/ directory and unzip the WFLW.zip
* Run SetPreparation.py to generate the training and test data.
　　
* By default, it repeats 10 times for every image for augmentation, and save results in ./data/test_data/ and ./data/train_data/ directory.
　　
## Training
Before training, check or modify network configuration (e.g. batch_size, epoch and steps etc..) in config.py.
  * MODEL_TYPE: you can choose PFLD, PFLD_Ultralight or PFLD_Ultralight_Slim for different network.
  * WIDTH_FACTOR: you can choose 1 for original network or 0.25 for narrower network.
  * TRAIN_DATA_PATH: the path of training data, by default it is ./data/train_data/list.txt which is generate by SetPreparation.py.
  * VAL_DATA_PATH: the path of validation data, by default it is ./data/test_data/list.txt which is generate by SetPreparation.py.

After modify the configuration, run train.py to start training.

## Test
Before test, modify the configuration in test.py, include the model_path, test_dataset, model_type, width_factor etc.
Then run test.py to align the images in the test dataset, and save results in ./test_result directory.

## C++ inference ncnn
1) Generate the onnx file: Modify the model_type, width_factor, model_path in pytorch2onnx.py, and then run pytorch2onnx.py to generate the xxx.onnx file.
2) Simplify the onnx file: First run "pip3 install onnx-simplifier" to install the tool, then run "python3 -m onnxsim xxx.onnx xxx_sim.onnx" to simpify the onnx file.
3) Transform onnx to ncnn format: Run "./onnx2ncnn xxx_sim.onnx xxx_sim.param xxx_sim.bin" to generate the ncnn files.

## References
* [PFLD](https://github.com/polarisZhao/PFLD-pytorch)
* [GhostNet](https://github.com/huawei-noah/ghostnet)

------


PFLD是适用于移动设备的高效的人脸关键点检测算法，GhostNet也是基于移动设备而生的高效的基础网络，下面是将两者进行结合后的优化网络，在不影响PFLD网络精度的前提下，将PFLD网络的效率进行大幅提升。

## > PFLD
[《PFLD: A Practical Facial Landmark Detector》](https://arxiv.org/pdf/1902.10859.pdf)是2019年3月腾讯、天津大学、武汉大学专门为移动设备联合推出的高效准确的人脸关键点检测算法。由于PFLD算法即优于SOTA的人脸关键点检测算法，又能够在移动设备高速运行，可以说PFLD算法是近几年中兼顾学术成果与工业应用的一个代表。在GitHub上也有TensorFlow和PyTorch的复现代码，大家可以去搜索试验一下PFLD的效果。

### PFLD网络结构
|Input|Operator|t|c|n|s|
|:----:|:----:|:----:|:----:|:----:|:----:|
|112x112x3|Conv3×3|-|64|1|2|
|56x56x64|DW Conv3×3|-|64|1|1|
|56x56x64|Inverted Residual Block|2|64|5|2|
|28x28x64|Inverted Residual Block|2|128|1|2|
|14x14x128|Inverted Residual Block|4|128|6|1|
|14x14x128|Inverted Residual Block|2|16|1|1|
|(S1) 14x14x16<br>(S2) 7x7x32<br>(S3) 1x1x128|Conv3×3<br>Conv7×7<br>-|-<br>-<br>-|32<br>128<br>128|1<br>1<br>1|2<br>1<br>-|
|S1,S2,S3|Full Connection|-|136|1|-|

### PFLD和GhostNet的结合体
PFLD和GhostNet的优化结合网络如下表：
| Input        | Operator   |t|c|n|s|
|:----:|:----:|:----:|:----:|:----:|:----:|
|112x112x3|Conv3×3|-|64|1|2|
|56x56x64|DW Conv3×3|-|64|1|1|
|56x56x64|GhostBottleneck|2|80|3|2|
|28x28x80|GhostBottleneck|3|96|3|2|
|14x14x96|GhostBottleneck|4|144|4|2|
|7x7x144|GhostBottleneck|2|16|1|1|
|7x7x16|Conv3×3|-|32|1|1|
|7x7x32|Conv7×7|-|128|1|1|
|(S1) 56x56x64<br>(S2) 28x28x80<br>(S3) 14x14x96<br>(S4) 7x7x144<br>(S5) 1x1x128|AvgPool<br>AvgPool<br>AvgPool<br>AvgPool<br>-|-<br>-<br>-<br>-<br>-|64<br>80<br>96<br>144<br>128|1<br>1<br>1<br>1<br>-|-<br>-<br>-<br>-<br>-|
|S1,S2,S3,S4,S5|Full Connection|-|136|1|-|

## > 优化结果
WFLW测试结果
|Model|Width|NME|Inference Time(NCNN 1xCore)|Model Size|
|:----:|:----:|:----:|:----:|:----:|
|PFLD|0.25<br>1|0.06163<br>0.05818|5.74ms<br>39.3ms|0.45M<br>5M|
|PFLD_Ghost|0.25<br>1|0.06072 (&darr;1.48%)<br>0.05652 (&darr;2.85%)|3.63ms (&darr;36.8%)<br>18.1ms   (&darr;53.9%)|0.41M (&darr;8.89%)<br>3.40M (&darr;32.0%)|

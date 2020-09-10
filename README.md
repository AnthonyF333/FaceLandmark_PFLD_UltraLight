这篇文章主要记录优化PFLD网络的实战经验，为需要优化网络模型的小伙伴们提供参考。优化后的网络在CPU上测试速度比原始网络提升了一倍以上，达到275fps，而且精度也有所提升。

> * PFLD简介
> * GhostNet简介
> * PFLD优化过程
> * 优化结果

------

## > PFLD

[《PFLD: A Practical Facial Landmark Detector》](https://arxiv.org/pdf/1902.10859.pdf)是2019年3月腾讯、天津大学、武汉大学专门为移动设备联合推出的高效准确的人脸关键点检测算法。由于PFLD算法即优于SOTA的人脸关键点检测算法，又能够在移动设备高速运行，可以说PFLD算法是近几年中兼顾学术成果与工业应用的一个代表。在GitHub上也有TensorFlow和PyTorch的复现代码，大家可以去搜索试验一下PFLD的效果。

### 1. PFLD网络结构
PFLD的基础网络是基于MobileNet V2进行修改的，而MobileNet V2的基础模块是Inverted Residual Block，正是Inverted Residual Block能够保持网络性能的同时，可以大大减少网络的参数、运算量，甚至是推理时间。PFLD的基础网络如下表：
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

可见PFLD基础网络中使用的是MobileNet V2的Inverted Residual Block，但是整体网络结构与MobileNet V2相比会有所差异，PFLD会比MobileNet V2更浅更窄，因此理论上PFLD比MobileNet V2更快速。
### 2. 损失函数
原始PFLD网络的训练用到了人脸姿态角度作为辅助信息，并和人脸关键点的误差结合起来作为最终的损失函数，有兴趣的小伙伴可以阅读一下PFLD的论文，这里不展开说明了。在实际训练过程中，尝试了只是应用Wing Loss而不加辅助信息来对网络进行训练，Wing Loss的最终测试效果是要优于原始PFLD加上辅助信息的损失函数的效果，因此后续的优化过程都是使用Wing Loss来进行训练的，没有使用PFLD的辅助信息。下面是Wing Loss的公式。
$$wingloss(x)= \begin{cases} \omega \, ln(1+|x|/\epsilon), & \text {if $|x|<\omega$} \\ |x|-C, & \text{otherwise} \end{cases} $$

## > GhostNet
这里简单介绍一下GhostNet，因为优化过程中会使用到GhostNet提出的GhostModule，而且正是GhostModule，PFLD的效率有了一个质的提升。
[《GhostNet: More Features from Cheap Operations》](https://arxiv.org/pdf/1911.11907.pdf)是2020年3月华为、北大为了提升CNN在嵌入式设备的运行效率而提出的算法。其中提出的GhostModule，巧妙地将PointWise Convolution和DepthWise Convolution结合起来，在达到相同通道输出的前提下，有效地减少运算量。虽然上面提到的Inverted Residual Block同样也是PointWise Convolution和DepthWise Convolution的结合体，但是两者将它们的结合方式有所不同，而正是GhostModule的巧妙结合，让网络的计算量得到减少。
其实按照个人的理解，GhostModule可以一句话概况为：输入的Tensor先做PointWise Convolution，再做DepthWise Convolution，最后将PointWise Convolution和DepthWise Convolution的结果按channel的维度进行拼接，作为最终的输出。大道至简，操作并不复杂，但却大大减少了运算量。

## > PFLD优化过程
众所周知，网络的精度和速度是鱼和熊掌不可兼得的关系，要提升精度必然损失速度，要提升速度则会损失精度。但在实际工程开发中，则希望两者兼得，老板最乐意看到你的模型又快又好，这就是社畜的宿命，社畜认命吧，那么要怎样做到呢？下面就详细介绍网络结构的优化过程。
### 1.调整PFLD网络结构，并增加多尺度全连层的数目（保持网络速度不变，尽可能提升网络精度）
对PFLD的基础网络进行调整，没有用NAS等技术，凭借经验和不断尝试尽量找出性能较好的网络结构，如果大家找到更好的网络结构，也欢迎大家分享出来。
此外，PFLD论文中提到 "To enlarge the receptive field and better catch the global structure on faces, a multi-scale fully-connected (MS-FC) layer is added for precisely localizing landmarks in images"，大概意思是为了增大感受野和更好地获取人脸的全局结构，多尺度全连层的嵌入，可以准确地定位人脸关键点。论文作者认为，多尺度全连层是为了提升网络的定位精度而加入的，因此在优化过程中，增加了多尺度全连层的数目，以提高网络的定位精度。
初步优化的网络结构如下：
| Input        | Operator   |  t  |c|n|s|
|:----:|:----:|:----:|:----:|:----:|:----:|
|112x112x3|Conv3×3|-|64|1|2|
|56x56x64|DW Conv3×3|-|64|1|1|
|56x56x64|Inverted Residual Block|2|64|3|2|
|28x28x64|Inverted Residual Block|3|96|3|2|
|14x14x96|Inverted Residual Block|4|144|4|2|
|7x7x144|Inverted Residual Block|2|16|1|1|
|7x7x16|Conv3×3|-|32|1|1|
|7x7x32|Conv7×7|-|128|1|1|
|(S1) 56x56x64<br>(S2) 28x28x64<br>(S3) 14x14x96<br>(S4) 7x7x144<br>(S5) 1x1x128|AvgPool<br>AvgPool<br>AvgPool<br>AvgPool<br>-|-<br>-<br>-<br>-<br>-|64<br>64<br>96<br>144<br>128|1<br>1<br>1<br>1<br>-|-<br>-<br>-<br>-<br>-|
|S1,S2,S3,S4,S5|Full Connection|-|136|1|-|
### 2.将Inverted Residual Block替换成GhostModule，并细调网络结构（保持网络精度不变，提升网络速度）
由于GhostModule的轻量化，将PFLD的Inverted Residual Block全部替换成GhostModule，并继续细调网络结构，最终优化的网络结构如下表：
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

## > To Be Continue
后续会将代码进行开源，欢迎大家来围观。


------

作者 [@Anthony Github][1]     
2020 年 09月 10日

[1]: https://github.com/AnthonyF333

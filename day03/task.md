### 7日打卡营 Class 3  - Homework
#### 作业形式
1. 本次作业为完整代码构建
2. 请同学们根据说明，写出完整的python代码。

#### 作业内容
##### 1. pspnet.py
- 根据课程和相关材料对pspnet的讲解，实现用Paddle动态图搭建pspnet
- PSPNet需要调用的backbone会提供给同学：`resnet_dilated.py`中实现了具有dilation的resnet。
- 建议同学使用`ResNet50`或者`ResNet101`为backbone网络。
   
##### 2. infer.py
- 完成模型预测代码。
- 该代码，输入为（1）训练好的分割模型（2）单张图像
- 输出为（1）分割模型的预测结果，numpy array格式用于测试性能（2）分割结果存为RGB图片（3）原图和预测彩色图叠加（overlay）的可视化图片
- 预测方式0：准备步骤，读图片，进行normalize等操作。
- 预测方式1：resize预测
- 预测方式2：划窗预测
- 预测方式3：划窗预测 + 随机翻转
- 预测方式4：多尺度 + 划窗预测 + 随机翻转

##### 3. resnet_dilated.py（选做）
- 自己实现ResNet，并按照PSPNet论文里的参数对标准ResNet进行修改
- 对`resnet.py`进行修改，实现dilated resnet。

#### 4. UNet实现（选做）
- 自己实现`UNet.py`,采用`Encoder-Decoder`结构
- 替换UNet现有的网络结构，例如，使用ResNet系列，或者Mobilenet系列。


##### 5. 提示：（可能需要做的事情）
  - 实现`PSPModule`类.
  - 实现`PSPNet`类，实现`__init__`和`forward`方法，初始化中创建DilatedResNet作为backbone网络，并且加入auxiliary loss。
  - 实现`main`函数，创建`PSPNet`对象进行简单的测试
#### Bonus：
- 根据前几节课的代码，整合数据读取、数据预处理、模型训练等模块, 进行PSPNet模型训练。选择若干图像对训练好的模型，进行测试，并进行可视化。

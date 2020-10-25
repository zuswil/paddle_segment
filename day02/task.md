7日打卡营 Class 2 - Homework

作业形式
  - 本次作业为代码"填空"和完整代码构建
 
  - 请同学们根据提供的python代码文件中，对相应的 **#TODO** 部分进行补全。

  - 请同学们根据说明，写出完整的python代码。

  - 每个python文件都可以直接运行，如果正确完成代码，并输出相关内容。

作业内容

basic_seg_loss.py

根据`basic_seg_loss.py`文件，补全程序并运行。

运行前分析可能得到的实验结果，并查看是否与程序运行结果一致。

**basic_train.py**

根据`basic_train.py`文件，补全程序并运行。

运行前分析可能得到的实验结果，并查看是否与程序运行结果一致。

**fcn8s.py**

根据`fcn8s.py`文件，补全程序并运行。

- Bonus：

根据Class1作业和本次作业，整合各个python文件，完成fcn8s在数据集上的训练工作。
可能需要做的事情：

  - 在`train.py`里修改路径等arguments

  - 在`train.py`里构建dataloader时增加数据增强操作

  - 在`train.py`里创建fcn8s对象

  - 在训练时增加val_dataloader, validation函数

  - 实现`iou`和`acc`并在`validation`时进行运算
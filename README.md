# CIFAR10分类 

本次实验使用的分类网络结构：六个卷积层，其中每两个卷积层后进行一次Maxpool层的采样，再通过6个卷积层后连接两个全连接层，最后连接softMax激活函数进行分类
本次实验所使用损失函数为crossentropy

### 1.包依赖:
#### python 3.8
#### numpy
    pip install numpy
#### pytorch 1.9.0
    pip install torch
#### torchvision version 0.2.1:
    pip install torchvision==0.2.1

<br/>

### 2.运行代码:
#### (1) 训练CIFAR10分类网络

    python train.py --dataset CIFAR10 --epochs 50
<br/>

### 3.网络结构分析

#### 特征提取层

(1) 两个卷积层带一个Maxpool层采样

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)
    
        self.Relu11 = nn.ReLU()
        self.features.append(self.Relu11)
    
        self.conv12 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv12)
    
        self.Relu12 = nn.ReLU()
        self.features.append(self.Relu12)
    
        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)

(2) 还是两个卷积层带一个Maxpool层采样

        self.conv21 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)
    
        self.Relu21 = nn.ReLU()
        self.features.append(self.Relu21)
    
        self.conv22 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv22)
    
        self.Relu22 = nn.ReLU()
        self.features.append(self.Relu22)
    
        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
(2) 最后两个卷积层带一个Maxpool层采样

        self.conv31 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv31)
     
        self.Relu31 = nn.ReLU()
        self.features.append(self.Relu31)
     
        self.conv32 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv32)
    
        self.Relu32 = nn.ReLU()
        self.features.append(self.Relu32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool3)

#### 分类器层

两个全连接层，第一个全连接层的激活函数使用Relu,第二个使用softmax进行分类，其中隐藏层的神经元个数为512，最后输出层为10

        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)
    
        # self.fc1act = nn.Sigmoid()
        self.fc1act = nn.ReLU()
        self.classifier.append(self.fc1act)
    
        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)

<br/>

### 4.实验结果分析

#### 试验参数说明

        learningRate：学习率，固定为1e-3
        epochs：训练轮数，固定为50
        BatchSize: 批次的大小，最开始为100，后面通过增加批次大小来观察这一超参数对于训练结果的影响

#### 实验结果
测试准确率：

        BatchSize = 100

![截屏2021-11-13 上午10.38.34](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/1.png)

```
    BatchSize = 200  
```

![截屏2021-11-13 上午10.51.44](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/2.png)

```
		BatchSize = 300
```

![截屏2021-11-13 上午10.53.29](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/3.png)

```
		BatchSize = 400
```

![截屏2021-11-13 上午10.54.20](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/4.png)

```
		BatchSize = 500
```

![截屏2021-11-13 上午10.55.29](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/5.png)

```
		BatchSize = 600
```

![截屏2021-11-13 上午10.57.08](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/6.png)

```
		BatchSize = 700
```

![截屏2021-11-13 上午10.58.24](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/7.png)

```
		BatchSize = 800
```

![截屏2021-11-13 上午10.59.02](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/8.png)

不同BatchSize的Loss与Accuracy的对比

Loss对比

![image-20211113111104176](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/Loss.png)

Accuracy对比

![image-20211113111558604](https://github.com/wljLlla/CIFAR10_Classification/blob/main/image/Accuracy.png)

可以发现，随着BatchSize的增长，收敛速度会上升，但是更容易陷入到局部最小值中，最后导致准确率不如小的BatchSize。

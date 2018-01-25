# cnn_handwritten_chinese_recognition
&nbsp;&nbsp;&nbsp;&nbsp; 使用`python+flask`搭建的一个网站，然后从网页的写字板上获取鼠标手写的汉字经过转码后传回后台，并经过图片裁剪处理之后传入`CNN`手写中文识别的模型中进行识别，最后通过`PIL`将识别结果生成图片，最后异步回传给web端进行识别结果展示。中文总共`50,000`多汉字，常用的有`3,755`个。这里主要对常见的`3755`个汉字进行识别。<br>
![demogif](https://github.com/taosir/cnn_handwritten_chinese_recognition/blob/master/cnn_handwrite_chinese_recognize.gif) <br>
## 一、数据集<br>
&nbsp;&nbsp;&nbsp;&nbsp; 目前国内有很多优秀的中文手写识别数据集。例如：北京邮电大学模式识别实验室发布的数据`(HCL2000)`，它是目前最大的脱机手写汉字库，共有`1,000`个人书写，除了汉字样本库外，还有一个对应的书写者信息库，记录了书写者的年龄、职业、文化程度等信息，用于研究相关影响因素。目前此数据库免费向研究者公开。本文使用的是中科院自动研究所的分享的中文手写数据集`CASIA-HWDB`(下载地址http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html )，由187个人通过`Wacom`笔在线输入的手写汉字。<br>
## 二、CNN结构：<br>
&nbsp;&nbsp;&nbsp;&nbsp; 用`tensorflow`库来实现【三个卷积层+三个池化层+两个全连接层】的卷积神经网络，结构如下图所示：<br>
![cnn_model_arch](https://github.com/taosir/cnn_handwritten_chinese_recognition/blob/master/cnn_handwrite_chinese_recognize_arch.png)
&nbsp;&nbsp;&nbsp;&nbsp; 训练图片都修整成64x64大小，这里我只训练常见的3755个汉字，在CNN识别数字的模型结构上再添加了一个卷积层和池化层，其他结构差不多。<br><br>
&nbsp;&nbsp;&nbsp;&nbsp; 将下载好的HWDB数据集解压处理好开始训练，这个训练过程比较长，我最终在`GPU:GTX1050Ti`上迭代了12,000次花费几个小时，最终取最可能的前三个预测值<br>
## 三、加载模型<br>
&nbsp;&nbsp;&nbsp;&nbsp; 我训练模型迭代了12,000次之后，将训练参数保存在`checkpoint`文件夹中，不过因为单个文件大小的限制，训练好的模型文件从百度云上下载：链接: https://pan.baidu.com/s/1eSWBIyI 密码: kv2r;<br><br>
&nbsp;&nbsp;&nbsp;&nbsp; 下载之后直接覆盖`checkpoint`文件夹。<br>
## 四、环境<br>
* python 3.6.1;<br>
* flask 0.12.2;<br>
* tensorflow 1.3.0;<br>
* pillow 4.2.1;<br>
* pickleshare 0.7.4;<br>
* numpy 1.13.1;<br>
## 五、运行
 1、克隆项目，然后按照环境要求安装好相应的库(使用`pip`安装)；<br>
 2、从百度云下载训练好的模型文件，放到相应的`checkpoint`文件夹下；<br>
 3、使用`python run.py`运行；<br>
 4、打开本地浏览器输入`localhost:5000`进行查看；<br>

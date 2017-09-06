# cnn_handwritten_chinese_recognition
  使用python+flask搭建的一个网站，然后从网页的写字板上获取鼠标手写的汉字经过转码后传回后台，并经过图片裁剪处理之后传入CNN手写中文识别的模型中进行识别，最后通过PIL将识别结果生成图片，最后异步回传给web端进行识别结果展示。
  中文总共50,000多汉字，常用的有3,755个。这里主要对常见的3755个汉字进行识别。
  ![demogif](https://github.com/taosir/cnn_handwritten_chinese_recognition/blob/master/cnn_handwrite_chinese_recognize.gif) 

一、数据集

  目前国内有很多优秀的中文手写识别数据集。例如：北京邮电大学模式识别实验室发布的数据(HCL2000)，它是目前最大的脱机手写汉字库，共有1,000个人书写，除了汉字样本库外，还有一个对应的书写者信息库，记录了书写者的年龄、职业、文化程度等信息，用于研究相关影响因素。目前此数据库免费向研究者公开。本文使用的是中科院自动研究所的分享的中文手写数据集CASIA-HWDB(下载地址http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html )，由187个人通过Wacom笔在线输入的手写汉字。

二、CNN结构：

  用tensorflow库来实现【三个卷积层+三个池化层+两个全连接层】的卷积神经网络，结构如下图所示：
  ![cnn_model_arch](https://github.com/taosir/cnn_handwritten_chinese_recognition/blob/master/cnn_handwrite_chinese_recognize_arch.png) 
  训练图片都修整成64x64大小，这里我只训练常见的3755个汉字，在CNN识别数字的模型结构上再添加了一个卷积层和池化层，其他结构差不多。
  将下载好的HWDB数据集解压处理好开始训练，这个训练过程比较长，我最终在GPU:GTX1050Ti上迭代了12,000次花费几个小时，最终取最可能的前三个预测值。
  
三、加载模型

  我训练模型迭代了12,000次之后，将训练参数保存在checkpoint文件夹中，不过因为单个文件大小的限制，训练好的模型文件从百度云上下载：
  链接: https://pan.baidu.com/s/1c1YT9tA 密码: j3gx;
  下载之后直接覆盖checkpoint文件夹。

四、环境

  python 3.6.1;

  flask 0.12.2;

  tensorflow 1.3.0;

  pillow 4.2.1;

  pickleshare 0.7.4;

  numpy 1.13.1;

五、运行

  1、克隆项目，然后按照环境要求安装好相应的库(使用pip安装)；

  2、从百度云下载训练好的模型文件，放到相应的checkpoint文件夹下；

  3、使用python run.py运行；

  4、打开本地浏览器输入localhost:5000进行查看；

  5、在线演示地址：http://119.23.253.113:5000/ 。



  
  
  


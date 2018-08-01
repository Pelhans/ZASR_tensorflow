# 基于tensorflow的语音识别系统
## 简介
基于 tensorflow 的中文语音识别框架, 模型结构参考 百度[Deepspeech2](http://proceedings.mlr.press/v48/amodei16.pdf) 的论文,训练语料采用Aishell 170 小时数据. 模型结构如下图所示:

![](img/arc.png)

## 运行

* 进入example/aishell 修改 run_data.sh 内相关存储路径后，运行该脚本生成 manfest.{train,dev,test} 文件 与 vocab.txt    
* 进入conf文件夹,修改 conf.ini 内的文件路径以及 hyparam.py 中的模型参数   
* 运行 train.py 训练模型    
* 运行test.py 进行测试

## server/client 运行

* 打开 ./demo_client.sh or ./demo_server.sh 文件配置 IP、端口等信息
* 执行 ./demo_server.sh 启动服务器    
* 执行 ./demo_client.sh 启动客户端   
* 在客户端内，持续按空格进行录音，松开空格后发送音频到服务器端进行语音识别

## TODO
* ~~根据论文修改模型结构~~    
* ~~实现 server/ client的调用~~  
* ~~输入MFCC特征更改为mfcc + 一阶差分 + 二阶差分~~    
* ~~输入改为语谱图~~    
* ~~全局使用 Batch Normalization~~    
* ~~结合语言模型进行解码~~

## Ref
以 [xxbb1234021/speech_recognition](https://github.com/xxbb1234021/speech_recognition) 和 [PaddlePaddle/Deepspeech2](https://github.com/PaddlePaddle/DeepSpeech)为基础进行修改

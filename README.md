# Huggingface Transformers实战教程

仅为本人代码整理及学习笔记

## 环境配置
1. 服务器个人账号路径下安装Anaconda
2. 创建新的虚拟环境
3. 安装 transformers（多种安装方式，建议固定一种，不要多种方式例如conda install、pip install都运行，否则会产生版本冲突，需要手动uninstall）
4. 运行```from transformers import AutoTokenizer, AutoModelForMaskedLM```测试即可
5. 使用vscode SSH Remote Connection远程修改代码

## 预训练模型下载方法
### 方法1：直接将预训练模型缓存到本地默认文件夹
1. 指定```PRETRAINED_MODEL_NAME = "bert-base-chinese"```，后面的名字可以是其他，在hugging face官网可以查到想要的模型名字，可以直接点击copy
![](https://s.readpaper.com/T/21F9wpWQ7ns)
2. ```tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)```
3. ```model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)```

默认文件夹在```.cache```隐藏文件夹内

### 方法2：先自定义路径克隆下来，再本地加载
1. ```git lfs install``` 下载克隆大文件所需的库
2. ```git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext```  后面的地址即官网某个模型的url
3. ```tokenizer = BertTokenizer.from_pretrained('C:\\Users\\yanqiang\\Desktop\\bert-base-chinese')```后面的地址即本地克隆的文件夹地址 
4. ```model = AutoModel.from_pretrained('本地路径')```


## Tokenizer不同函数的对比
![](https://s.readpaper.com/T/21FET5EJeEM)
![](https://s.readpaper.com/T/21FEYsW0J84)

## BERT模型返回值：
1.	```last_hidden_state```：**模型最后一层输出的隐藏状态**
2.	```pooler_output```：**序列的第一个token：[CLS]的最后一层的隐藏状态**
3.	```(hidden_states)可选```：需要指定```config.output_hidden_states=True```。是一个元组，第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)。
4.	```(attentions)可选```：需要指定```config.output_attentions=True```。是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。

代码实战中，只将```pooler_output```的结果即[CLS]作为输出，输入到最后的分类层中来完成分类任务，可能效果不太好。可供改进的方案有(输出的选择)：
1.	求```last_hidden_state```的向量平均，即最后一层隐藏状态的平均
2.	取出```hidden_states```最后四层，然后做平均
3.	用[CLS]的输出```last_hidden_state + LSTM```，将时序信息考虑进来


## 不同任务的模型汇总
1. ```AutoModel```：一般需要自定义模型，例如最后再接一个全连接层做分类任务
2. ```AutoModelForSequenceClassification```：为每个文本输出一个分类logits，若同时输入真实labels，还会返回loss
3. ```BertForTokenClassification```：为每个token输出一个分类logits
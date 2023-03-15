# 04-句子相似性识别实战

## 1. 数据集介绍
Microsoft Research Paraphrase Corpus (MRPC)，属于GLUE基准测试的一部分：**有两个句子，您想预测一个句子是否是另一个句子的释义。评价指标为 F1 和准确率。**

另：[中文语言理解测评基准(CLUE)](https://www.cluebenchmarks.com/classification.html)

## 2.构建数据集

![](https://s.readpaper.com/T/21EfoSM9ule)
1. 一定要在```CustomDataset```自定义数据集类的```__init__()```函数中就```self.tokenizer = AutoTokenizer.from_pretrained(bert_model)```来载入分词器，**不要在```__getitem___()```中载入，否则每次索引都会调用载入，特别费时！**
2. ```tokenizer```也可以接收两个句子，因为我们要分析两个句子之间的相似性。后续```token_type_ids```会帮助我们区分两个句子，0代表第一个句子，1代表第一个句子。
3. ```padding=```用于填充,设置为```True```或```longest```表示填充到最长序列，**但如果输入的句子只有一个时，不会填充！**一般设置为```max_length```，即根据```max_length```参数填充。

**Dataset创建好之后要封装成dataloader**

## 3. 构建模型（SentencePairClassifier）
1. 不同模型的隐藏层大小不同，所以最后全连接层(分类层)的参数设置也不同
2. 可以冻结BERT预训练模型参数，即只更新下游任务的超参数，加快训练速度
3. ```@autocast()```可以自动使用混合精度训练，节省显存，加快训练速度
```python
class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  初始化预训练模型Bert xxx
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  encoder 隐藏层大小
        if bert_model == "albert-base-v2":  # 12M 参数
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M 参数
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M 参数
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M 参数
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M 参数
            hidden_size = 768
        elif bert_model == "roberta-base": # 
            hidden_size = 768

        # 固定Bert层 更新分类输出层
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
                
        self.dropout = nn.Dropout(p=0.1)
        # 分类输出
        self.cls_layer = nn.Linear(hidden_size, 1)


    @autocast()  # 混合精度训练
    def forward(self, input_ids, attn_masks, token_type_ids):
        outputs = self.bert_layer(input_ids, attn_masks, token_type_ids)
        logits = self.cls_layer(self.dropout(outputs['pooler_output']))

        return logits
```


## 4. 加速GPU训练方法
GPU的性能主要分为两部分：算力和显存，前者决定了显卡计算的速度，后者则决定了显卡可以同时放入多少数据用于计算。在可以使用的显存数量一定的情况下，每次训练能够加载的数据更多（也就是batch size更大），则可以提高训练效率。

### 1. amp.autocast() + GradScaler()
Pytorch默认的浮点数存储方式用的是```torch.float32```，但大多数场景我们不需要数据如此精确，我们只牺牲一半的精度也就是改为半精度```torch.float16```格式，能够减少显存占用，使得显卡可以同时加载更多数据进行计算。             

autocast自动应用精度到不同的操作。因为损失和梯度是按照float16精度计算的，当它们太小时，梯度可能会“下溢”并变成零。

GradScaler通过将损失乘以一个比例因子来防止下溢，根据比例损失计算梯度，然后在优化器更新权重之前取消梯度的比例。如果缩放因子太大或太小，并导致inf或NaN，则缩放因子将在下一个迭代中更新缩放因子。

使用方法见代码```train_bert()```函数

### 2. 梯度累加
每次获取1个batch的数据，计算1次梯度，梯度不清空，不断累加。累加一定次数后，根据累加的梯度更新网络参数，然后清空梯度，进行下一次循环。

一次算的batchsize越大，越消耗显存。而梯度累加，每次算一个小的batchsize，但是一个大的batchsize才更新一次梯度，约等于使用了一个大的batchsize。

详解见[pytorch中如何做梯度累加](https://zhuanlan.zhihu.com/p/351999133)

### 3. nn.DataParallel
使用```nn.DataParallel```在多个显卡上并行运算，分摊显存，但目前经过测试会降低速度！



## 附：tqdm库
![](https://s.readpaper.com/T/21EphvljUG2)
![](https://s.readpaper.com/T/21EpmyGO6Vc)
1.	```tqdm()```括号中必须是一个可迭代对象，例如range或dataloader
2.	一旦调用```print()```就会打印当前进度条
3.	可设置```tqdm(iter, desc='Processing')```给进度条添加描述
4.	输出内容```00:01<00:00```表示当前花了00:01还需00:00，以及每秒迭代数


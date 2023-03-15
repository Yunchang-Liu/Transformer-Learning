# 03-文本多标签分类实战

## 1. 数据集介绍
推特Toxic Comment多标签分类：一共六列标签，1代表属于这类标签，0代表不属于这类标签。***一个文本可以有多个标签。***

![](https://s.readpaper.com/T/21ErsnwYfGC)

## 2.数据预处理
![](https://s.readpaper.com/T/21Et5hKnaxF)
![](https://s.readpaper.com/T/21EsjDeatk1)

我们还能发现，不同文本的标签分布极其不均匀，例如属于threat类的文本特别少。

## 3. 构建数据集
分词时采用```batch_encode_plus()```函数，将批量文本一次性分词，而不是一个一个分词。
![](https://s.readpaper.com/T/21Ew8MfcKCf)

由于标签分类不均匀，在```train_test_split```考虑设置```stratify=labels```,从而保证训练集和测试集会按照y的比例分配。但也要注意，不能出现某个独热编码只有1个样本的情况，否则无法均匀分配到训练集和测试集。
![](https://s.readpaper.com/T/21Evy2YOqdV)

## 4. 加载模型（xxxModelForSequenceClassification）
```num_labels```指定模型输出几个类别，可以让模型自动设置最后一个Linear层的输出维度。

![](https://s.readpaper.com/T/21Ex0zeh8mv)
```xxxModelForSequenceClassification```模型的输出有两个：
1. loss
2. logits

但注意：如果给```model()```的参数中有真实标签，那么模型的输出有两个loss和logits，**如果输入的参数没有真实标签，模型的输出只有logits**

## 附：pandas的str列内置方法
在使用pandas框架的DataFrame的过程中，如果需要处理一些字符串的特性，例如判断某列是否包含一些关键字，某列的字符长度是否小于3等等这种需求，如果掌握str列内置的方法，处理起来会方便很多。

### 例：统计文本word个数大于200的文本个数
```python
    sum(df.comment_text.str.split().str.len()>200) # 统计文本word个数大于200的文本个数
```
1. ```df.comment_text.str.split()```对df的comment_text这一列的每一行都进行```split()```操作。返回值是一列数据，其中每行都是原字符串```split()```后的列表
2. 后面的```.str.len()```是再统计上一步列表的长度，任务中是为了统计每个评论有多少个单词
3. ```>200```让此列变成了布尔型，1代表大于200，0代表不大于200
4. ```sum()```将所有的1求和，即求所有的文本中共有多少大于200


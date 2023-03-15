# 05-命名实体识别实战

## 1. 数据预处理
训练数据集中，每一行由 **[文字+空格+实体标注]** 组成，两句话之间由**空行**分割
![训练数据](https://s.readpaper.com/T/21AVBmUSpaz)
处理时要注意文中**存在空白字符**
### 目标效果：
![目标效果](https://s.readpaper.com/T/21AWJHGWxpQ)

## 2.构建dataloader
自定义Dataset步骤：
1. 对每个句子分词：使用自定义的```tokenize_and_preserve_labels```函数，其中使用了```tokenizer.tokenize()```函数对单个词进行分词。**由于是单个词进行调用，所以后续需要手动添加[CLS]等token，以及要手动填充、构建masks等BERT的输入参数。**
2. 添加特殊token并添加对应的标签
3. 截断/填充
4. 构建attention mask
5. 将分词结果转为词表的id表示

![](https://s.readpaper.com/T/21AXPK9vdgn)

**Dataset创建好之后要封装成dataloader**

## 3. 模型定义（BertForTokenClassification）
```python
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labels_to_ids))
    model.to(device)
```

注意事项：
1. BERT模型的输入是都是(batch_size, sequence_length)，即**二维张量**；如果准备的输入是一维张量，需要```unsqueeze(0)```方法
2. 模型的输出有两个：一个为loss和一个为logits；logits维度为 (batch_size, sequence_length, num_labels)

![](https://s.readpaper.com/T/21AYiQzIO2b)


## 4. 计算准确率
1. 将一个批次的真实标签(二维张量)展平成一维向量
2. 将模型输出的概率logits展为batch_size * seq_len个（n个分类分别的概率值）
3. 使用```argmax```将输出结果中概率最大的当做预测结果
4. 句子所有的[PAD]有固定标签"O"，所以无需计算准确率
5. 利用attention_masks和```torch.masked_select```方法将[PAD]过滤
6. 使用```accuracy_score```函数比较真实值和预测值获得准确率


```python
    outputs = model(input_ids, attention_masks, labels)
    logits = outputs[1]

    # 计算准确率
    flattened_labels = labels.view(-1)  # 本来是二维: batch个labels组成的列表,展平成一维(batch_size * seq_len)
    active_logits = logits.view(-1, model.num_labels) # 模型输出shape (batch_size * seq_len, num_labels)
    flattened_logits = torch.argmax(active_logits, axis=1)

    # MASK所有的[PAD]
    activate_accuracy = attention_masks.view(-1) == 1
    targets = torch.masked_select(flattened_labels, activate_accuracy)
    predictions = torch.masked_select(flattened_logits, activate_accuracy)

    tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
    tr_accuracy += tmp_tr_accuracy
```

## 附：函数对比
### 1. view()和squeeze()
- ```view()```：可以设置view(-1)自动计算调整后的维度，随意调整维度
- ```squeeze()```：只能将某维度为1的给挤压掉，实现降维

### 2. append()和extend()和+[]
- ```append(x)```：向列表末尾添加元素
- ```extend(x)```：参数是可迭代对象，可以理解成**向列表末尾逐个添加迭代元素**
- ```+[]```：可以拼接列表，例如["CLS"] + ["hhh"] + ["SEP"] = ["CLS","hhh","SEP"]

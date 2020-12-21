# ChineseNMT

Homework2 of Computational Linguistics -- NMT(en-ch)

## Data Process

### 分词

- 工具：sentencepiece包
- 预处理：`./data/get_corpus.py`抽取train、dev和test中双语语料，分别保存到`corpus.en`和`corpus.ch`中，每行一个句子。
- 训练分词模型：`./tokenizer/tokenize.py`中调用了sentencepiece.SentencePieceTrainer.Train()方法，利用`corpus.en`和`corpus.ch`中的语料训练分词模型，训练完成后会在`./tokenizer`文件夹下生成`chn.model`，`chn.vocab`，`eng.model`和`eng.vocab`，其中`.model`和`.vocab`分别为模型文件和对应的词表。

## Model

采用harvard开源的 [transformer-pytorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html) ，中文说明可参考 [传送门](https://zhuanlan.zhihu.com/p/144825330) 。

# Usage

模型参数在`config.py`中设置。

- 由于transformer显存要求，支持MultiGPU，需要设置`config.py`中的`device_id`列表以及`main.py`中的`os.environ['CUDA_VISIBLE_DEVICES']`。

## Results

Before：

1. 使用包装后的Adam优化器NoamOpt，测试集bleu：**0.77**

2. 测试结果（idx:序号 gold translation ||| prediction）

   ```
   ......
   idx:9988此次衰退已经延续了五年，而且仍不会马上结束。|||当前衰退已经持续了半年的水平,不会很快结束。
   idx:9989而恐怖组织通常不会与具体国家的政府有联系。|||恐怖组织通常没有与政府关系。
   idx:9990还有更多事情需要完成，特别是在国内资源动员方面。|||还需要做更多的事,特别是在国内资源动员方面。
   idx:9991这可不是G-20——或其他任何人——在2009年想要的结果。|||这不是G-20——或任何其他任何人——在2009年都试图重回。
   idx:9992但保持这一势头将是美联储在未来一年中的主要挑战。|||但在未来一年中,美联储面临重大挑战。
   idx:9993他与陷入丑闻的古普塔（Gupta）家族的个人关系将十分碍眼。|||他与危地马拉家族的丑闻联系将集中于此。
   idx:9994&lt;}0{&gt;免债鼓吹者们援引长期形成的国际法理论，宣称伊拉克债务是“可憎”的。 因此，债主们不再受到国际法律规则的保护。|||结果,债权人不再受到全球法律规则的保护。
   idx:9995埃博拉危机还提醒我们政府和公民社会的重要性。|||危机还提醒我们政府和公民社会的重要性。
   idx:9996与此同时，群众不再愿意接受地区和全球机构。|||与此同时,民众拒绝地区和全球机构。
   idx:9997但这次人们却倍感挫败，因为德国公然地拒绝领军。|||但他们看到的创伤,因为德国公开拒绝领导责任。
   idx:9998设计正确的公共创投资本制度是困难的。|||设计公共风险资本的正确机构很难。
   idx:9999美国不但鼓励这些变化，也从中获得了巨大的收益。|||美国不仅鼓励这些改变,而且从他们身上受益匪浅。
   ......
   ```

After：

| Model | NoamOpt | LabelSmoothing | Best Dev Bleu | Test Bleu |
| :---: | :-----: | :------------: | :-----------: | :-------: |
|   1   |   No    |       No       |     22.78     |   22.65   |
|   2   |   Yes   |       No       |               |           |
|   3   |   No    |      Yes       |               |           |
|   4   |   Yes   |      Yes       |               |           |



## TODO

- 尝试改进模型
- 测试：optimizer, LabelSmoothing等对结果的提升效果。
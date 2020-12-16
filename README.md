# ChineseNMT

Homework2 of Computational Linguistics -- NMT(en-ch)

## 数据处理

### 分词

- 工具：sentencepiece包
- 预处理：`./data/get_corpus.py`抽取train、dev和test中双语语料，分别保存到`corpus.en`和`corpus.ch`中，每行一个句子。
- 训练分词模型：`./tokenizer/tokenize.py`中调用了sentencepiece.SentencePieceTrainer.Train()方法，利用`corpus.en`和`corpus.ch`中的语料训练分词模型，训练完成后会在`./tokenizer`文件夹下生成`chn.model`，`chn.vocab`，`eng.model`和`eng.vocab`，其中`.model`和`.vocab`分别为模型文件和对应的词表。

## 模型

采用harvard开源的 [transformer-pytorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html) ，中文说明可参考 [传送门](https://zhuanlan.zhihu.com/p/144825330) 。

## TODO

- greedy_decode的batch实现。
- 超参数调整：学习率等等。
import torch

import logging
import sacrebleu
from tqdm import tqdm

import config
from model import greedy_decode
from model import batch_greedy_decode
from utils import chinese_tokenizer_load


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, criterion, optimizer):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model, LossCompute(model.generator, criterion, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()
        dev_loss = run_epoch(dev_data, model, LossCompute(model.generator, criterion, None))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, oDev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            logging.info("-------- Save Best Model! --------")


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            #self.opt.zero_grad()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


def evaluate(data, model):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 待翻译的英文句子
            en_sent = batch.src_text
            # 对应的中文句子
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            decode_result = batch_greedy_decode(model, src, src_mask,
                                               max_len=config.max_len)
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
            # print(cn_sent[0], translation[0])
            # 打印模型翻译输出的中文句子结果
            # for i in range(len(en_sent)):
            #     src = batch.src[i]
            #     # 增加一维
            #     src = src.unsqueeze(0)
            #     # 设置attention mask
            #     src_mask = (src != 0).unsqueeze(-2)
            #     # 用训练好的模型进行decode预测
            #     decode_result = greedy_decode(model, src, src_mask,
            #                                   max_len=config.max_len).squeeze().tolist()
            #     # 模型翻译结果解码
            #     translation = sp_chn.decode_ids(decode_result)
            #     trg.append(cn_sent[i])
            #     res.append(translation)
            #     if i == 3:
            #         break
    res = [res]
    bleu = sacrebleu.corpus_bleu(trg, res)
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        # 开始预测
        bleu_score = evaluate(data, model)
        # test_loss = run_epoch(data, model, LossCompute(model.generator, criterion, None))
        test_loss = "None"
        logging.info('Test loss: {}, Bleu Score: {}'.format(test_loss, bleu_score))

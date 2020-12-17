import torch

d_model = 256
n_heads = 4
n_layers = 3
d_k = 32
d_v = 32
d_ff = 1024
dropout = 0.1
padding_idx = 0
src_vocab_size = 10000
tgt_vocab_size = 10000
batch_size = 32
epoch_num = 20
lr = 3e-4
share_enc_dec_weights = False
share_dec_proj_weights = True

# greed decode的最大句子长度
max_len = 60

data_dir = './data'
train_data_path = './data/train.json'
dev_data_path = './data/dev.json'
test_data_path = './data/test.json'
model_path = './experiment/model.pth'
log_path = './experiment/train.log'

gpu = '3'

# set device
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device('cpu')

import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 8
epoch_num = 5
lr = 3e-4
share_enc_dec_weights = False
share_dec_proj_weights = True

data_dir = './data'
train_data_path = './data/train.json'
dev_data_path = './data/dev.json'
test_data_path = './data/test.json'
model_path = './experiment/'
log_path = './experiment/train.log'

gpu = '0'

# set device
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device('cpu')

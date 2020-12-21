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
batch_size = 32
epoch_num = 30
early_stop = 5
lr = 3e-4

# greed decode的最大句子长度
max_len = 60

data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'
model_path = './experiment/model.pth'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'

gpu_id = '0'
device_id = [0, 1, 2, 3]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
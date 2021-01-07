import config
import torch
from torch.utils.data import DataLoader
from train import train, test
from data_loader import MTDataset
from model import make_model

if __name__ == "__main__":
	test_dataset = MTDataset(config.test_data_path)
	test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
	                             collate_fn=test_dataset.collate_fn)
	# 初始化模型
	model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
	                   config.d_model, config.d_ff, config.n_heads, config.dropout)
	model_par = torch.nn.DataParallel(model)
	# 训练
	criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
	test(test_dataloader, model, criterion)
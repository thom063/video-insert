import argparse

import torch
from dataset.vimeo90k import get_loader
from model.core.cain_noca import CAIN_NoCA
from model.train import ModelTrain
from torch.optim import Adam
import logging
from distribute_train_struct.master_node.master_manager import MasterManager

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s -%(message)s')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--master_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--data_root', default=r'./data/vimeo_septuplet', type=str, required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False)
    parser.add_argument('--lr', default=1e-4, type=float, required=False)
    args = parser.parse_args()

    worker_cache_path = args.worker_cache_path
    master_cache_path = args.master_cache_path
    batch_size = args.batch_size
    data_root = args.data_root
    lr = args.lr

    model = CAIN_NoCA(3)
    optimizer = Adam(model.parameters(), lr=lr)
    m = MasterManager(model, optimizer, worker_cache_path, master_cache_path)
    m.monitor2(1)

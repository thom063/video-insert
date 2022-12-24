import argparse
import logging

import torch
from distribute_train_struct.worker_node.worker_manager import WorkerManager

from dataset.data_parallel import DataParallel
from dataset.vimeo_block import get_loader
from model.core.cain_noca import CAIN_NoCA
from model.train import ModelTrain

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s -%(message)s')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--master_cache_path', default='./cache', type=str, required=False)
    parser.add_argument('--data_root', default=r'D:\Git\ai-inter\data\vimeo_septuplet', type=str, required=False)
    parser.add_argument('--batch_size', default=2, type=int, required=False)
    parser.add_argument('--device', default='cpu', type=str, required=False)
    parser.add_argument('--lr', default=1e-4, type=float, required=False)
    args = parser.parse_args()

    worker_cache_path = args.worker_cache_path
    master_cache_path = args.master_cache_path
    batch_size = args.batch_size
    data_root = args.data_root
    device = args.device
    lr = args.lr

    model = CAIN_NoCA(3)
    model.to(torch.device(device))
    data_loader = get_loader('train', data_root, batch_size, shuffle=True)
    data_loader = DataParallel(data_loader)
    train = ModelTrain(model, data_loader.data_consumer(), None, lr=lr, device=torch.device(device))
    # 配置worker
    worker = WorkerManager(model, train.optimizer, worker_cache_path, master_cache_path)
    train.set_worker_manage(worker)
    train.train()
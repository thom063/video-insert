import logging

import torch
from distribute_train_struct.worker_node.worker_manager import WorkerManager
from torch import nn
from torch.utils.data import DataLoader

from dataset.vimeo90k import get_loader
from model.core.cain_noca import CAIN_NoCA
from torch.optim import Adam

from model.tools.model_test import test, AverageMeter
from model.tools.tools import save_checkpoint


class ModelTrain():
    lr: float
    epoch: int
    log_iter: int
    model: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    worker_manage: WorkerManager = None
    # 测试的间隔
    test_epoch: int = 2
    device: torch.device

    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 epoch=100, lr=1e-4,
                 log_iter=500, device: torch.device = torch.device("cpu")):
        self.log_iter = log_iter
        self.lr = lr
        self.epoch = epoch
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.device = device

    def set_worker_manage(self, worker: WorkerManager):
        self.worker_manage = worker

    def train(self):
        # 梯度管理器
        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        # 损失函数
        loss_function = nn.L1Loss()

        best_psnr = 0
        for e in range(self.epoch):
            losses = AverageMeter()
            for i, (inputs, gt) in enumerate(self.train_loader):
                im1 = inputs[0].to(self.device)
                im2 = inputs[1].to(self.device)
                gt = gt.to(self.device)

                loss_function.train()
                out, feats = self.model(im1, im2)
                loss = loss_function(out, gt)

                # 更新模型
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item())

                if i % self.log_iter == 0:
                    logging.info('Train Epoch: {} [{}/{}] Loss: {:.6f})'.format(
                        self.epoch, i, len(self.train_loader), losses.avg))
            # 广播变量
            if self.worker_manage is not None:
                self.worker_manage.broadcast_model()

            if self.test_loader is not None:
                test_loss, psnr, _, _ = test(self.model, loss_function, self.test_loader, 'L1', e)
                # save checkpoint
                is_best = psnr > best_psnr
                best_psnr = max(psnr, best_psnr)
            else:
                is_best = True
            save_checkpoint(self.model.state_dict(), is_best, "exp")
            # 更新学习率
            scheduler.step()

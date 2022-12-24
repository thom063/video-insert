r"""数据多线程处理
"""
from queue import Queue

from torch.utils.data import DataLoader

from model.tools.base_tools import thread


class DataParallel:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        # 队列设置
        self.data_queue = Queue(10)
        self.task_end = False
        self.data_producer()
    @thread(True)
    def data_producer(self):
        """
        数据生产者
        """
        for data in self.dataloader:
            self.data_queue.put(data)
        self.task_end = True

    def data_consumer(self):
        while 1:
            if not self.task_end:
                yield self.data_queue.get()
            else:
                break

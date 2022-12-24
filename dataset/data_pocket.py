import random
from typing import List
import numpy as np


class DataPocket:
    _sample_1: List[np.ndarray]

    def __init__(self, data_list: List[np.ndarray]):
        """

        :param data_list:
        """
        # package_len = len(data_list)
        # label_index = random.choice(list(range(package_len)[1: package_len - 1]))
        self._sample_1 = data_list[:3]

    #
    # def get_data_list(self):
    #     return self._data_list
    #
    # def get_train_data(self):
    #     package_len = len(self._data_list)
    #     label_index = random.choice(list(range(package_len)[1: package_len-1]))
    #     return self._data_list[label_index-1:label_index+1]

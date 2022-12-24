import os
import shutil

import torch


def save_checkpoint(state, is_best, exp_name, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "checkpoint/%s/" % (exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoint/%s/' % (exp_name) + 'model_best.pth')

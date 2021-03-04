import numpy as np
import torch


def init_device(use_gpu: bool = True):
    """
    Method to set the device for pytorch
    :param use_gpu:
    :return:
    """
    if torch.cuda.is_available() and use_gpu:
        dev = "cuda:0"
    else:
        dev = "cpu"

    print(f"Using device: {torch.device(dev)}")
    return dev


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

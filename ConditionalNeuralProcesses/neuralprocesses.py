# -*- coding: utf-8 -*-

##
import random
import time

import numpy as np
import torch
from IPython.display import clear_output
from dataset import get_training_data
from ConditionalNeuralProcesses.models import NeuralProcess
from plot_helper import Plotter
from utils import init_device

torch.autograd.set_detect_anomaly(True)

##
# @title Hyperparameter setting
USE_GPU = True  # @param {type:"boolean"}
LATENT_SPACE = 128  # @param {type:"slider", min:0, max:1000, step:1}
HIDDEN_DIMS = 128  # @param {type:"slider", min:0, max:1000, step:32}
BATCH_SIZE = 16  # @param {type:"slider", min:16, max:1000, step:16}
NUMBER_POINTS = 200  # @param {type:"slider", min:0, max:200, step:10}
SEED = 2021  # @param {type:"string"}
EPOCHS = 3000021  # @param {type:"integer"}
WINDOW_SIZE_PLOTTING = 2000  # @param {type:"slider", min:0, max:50000, step:1000}
HIDDEN_LAYER_ENCODER = 1  # @param {type:"slider", min:0, max:10, step:1}
HIDDEN_LAYER_DECODER = 2  # @param {type:"slider", min:0, max:10, step:1}

dev = init_device(USE_GPU)
np.random.seed(2021)

##
plotter = Plotter()

process = NeuralProcess(hidden_dims=HIDDEN_DIMS,
                        hidden_layer_encoder=HIDDEN_LAYER_ENCODER,
                        hidden_layers_decoder=HIDDEN_LAYER_DECODER,
                        lr=3e-4)
process.to(dev)

loss_eval, epoch_eval = [], []

# Test Curve
test_c_points, test_t_points_X, _, test_data = get_training_data(number_points=NUMBER_POINTS, batch_size=1,
                                                                 context_points=10)
test_c_points = torch.Tensor(test_c_points).to(dev)
test_t_points_X = torch.Tensor(test_t_points_X).to(dev)

##
for i in range(EPOCHS):

    tic = time.time()
    number_context = random.randint(3, 10)

    c_points, t_points_X, t_points_y, data = get_training_data(number_points=NUMBER_POINTS, batch_size=BATCH_SIZE,
                                                               context_points=number_context)

    c_points = torch.Tensor(c_points).to(dev)
    t_points_X = torch.Tensor(t_points_X).to(dev)
    t_points_y = torch.Tensor(t_points_y).to(dev)

    loss, mu, log_sigma, metrics, z = process.train_process(c_points, t_points_X, t_points_y)

    loss_eval.append(loss.detach().cpu().numpy())
    epoch_eval.append(i)

    toc = time.time()
    if i % 100 == 0:
        print(f"EPISODE {i}: {torch.mean(loss.detach()):.4f} | {toc-tic:.4}s")

    if i % 2000 == 0:
        process.eval()
        _, test_mu, test_log_sigma, test_metrics, test_z = process(test_c_points,
                                                                   test_t_points_X,
                                                                   None)
        clear_output(wait=True)

        train_curve = {
            "data": data,
            "mu": mu.detach()[0].cpu().numpy(),
            "log_sigma": log_sigma.detach()[0].exp().cpu().numpy(),
            "z": z.cpu().detach().numpy(),
            "c_points": c_points.cpu().numpy()[0]
        }

        test_curve = {
            "data": test_data,
            "mu": test_mu.detach()[0].cpu().numpy(),
            "log_sigma": test_log_sigma.detach()[0].exp().cpu().numpy(),
            "z": test_z.cpu().detach().numpy(),
            "c_points": test_c_points[0].cpu().numpy()
        }
        plotter.plot_process(i, loss_eval, epoch_eval, train_curve=train_curve, test_curve=test_curve)

    if i % 350001 == 0 and i > 1:
        plotter.save_vid(f"test{i}.mp4")

##
if __name__ == '__main__':
    pass

##

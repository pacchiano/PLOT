import numpy as np
import torch
import IPython

from algorithms import train_baseline


dataset = "Bank-10_10"
num_timesteps = 10000
batch_size = 32
representation_layer_sizes = [10,10]



accuracy_info, baseline_model =  train_baseline(dataset, num_timesteps, batch_size, 
    representation_layer_sizes = representation_layer_sizes)



IPython.embed()
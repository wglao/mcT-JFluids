# sets parameters for use in Generate_data and Training scripts
import numpy as np
from jax import random

# random seed for training data
key_data_train_a = random.PRNGKey(1)
key_data_train_b = random.PRNGKey(2)

# random seed for test data
key_data_test_a = random.PRNGKey(3)
key_data_test_b = random.PRNGKey(4)

key_data_noise = random.PRNGKey(5)

# Generate_data_initilizers
num_train_samples = 100
num_test_samples = 100

# Training inputs
num_train = 100
num_test = 100


n_seq = 1
n_seq_mc = 1

noise_level = 0.02
mc_alpha = 1e5
learning_rate = 1e-4
layers = 1
batch_size = 40

units = 5000

num_epochs = int(1.5e5)

facdt = 1

u = 0
v = 0
w = 0

c = 0.9
T = 0.2
Nt = 1000
dt = T / Nt
nt_train_data = 100
nt_test_data = 1000
x_max = 1.0
dx = dt / c
N = int(x_max / dx)


n_plot = 3
# Plot_Steps = [0, 50, 100, 200, 500]
Plot_Steps = np.linspace(0, nt_test_data, n_plot, dtype=int)

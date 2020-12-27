import data as dt
import pso as ps
import neural_network as nn


# Data preprocessing
dt.first_data_handle()
dt.products_with_sales()
dt.create_week_data()
dt.create_nn_data()
dt.pso_data()


# Test the neural network
nn.nn_testing()


# Optimize prices using particle swarm optimization
ps.optimize_prices()

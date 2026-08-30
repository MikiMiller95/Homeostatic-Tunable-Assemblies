"""Weight-matrix helpers used by the Figure 2, row 3 spiking simulation."""

import matplotlib.pyplot as plt
import numpy as np


def generate_bernouilli_wx(init_weights, N, N_X, num_per_pool):
    """Generate the optional sparse external weight matrix."""
    p = 0.1
    weights = np.zeros(shape=(N, N_X // num_per_pool))
    num = N_X // num_per_pool

    weights[:N // 2, :num // 2] = (np.random.rand(N // 2, num // 2) < p) * init_weights[0]
    weights[:N // 2, num // 2:] = 0
    weights[N // 2:, :num // 2] = 0
    weights[N // 2:, num // 2:] = (np.random.rand(num // 2, N // 2) < p) * init_weights[1]
    return weights, p


def generate_bernouilli_weight_array(init_weights, N_E, N_I):
    """Generate the optional sparse recurrent weight matrix."""
    p = 0.1
    weights = np.zeros(shape=(N_E + N_I, N_E + N_I))

    w_EE = init_weights[0, 0]
    weights[:N_E, :N_E] = (np.random.rand(N_E, N_E) < p) * w_EE
    w_EI = init_weights[0, 1]
    weights[:N_E, N_E:] = (np.random.rand(N_E, N_I) < p) * w_EI
    w_IE = init_weights[1, 0]
    weights[N_E:, :N_E] = (np.random.rand(N_I, N_E) < p) * w_IE
    w_II = init_weights[1, 1]
    weights[:N_I, :N_I] = (np.random.rand(N_I, N_I) < p) * w_II
    return weights


def generate_weight_array(init_weights, num_E, num_I):
    """Generate the dense recurrent weight matrix used in the simulation."""
    variances = [.00000005, .00000005, .00000005, .00000005]
    row_num = [num_E, num_I]
    Num_neur = num_E + num_I
    rep_weights = np.zeros(shape=(num_E + num_I, num_E + num_I))
    shapes = [(num_E, num_E), (num_E, num_I), (num_I, num_E), (num_I, num_I)]
    ct = 0

    for i in range(init_weights.shape[0]):
        for j in range(init_weights.shape[1]):
            variance = variances[ct]
            mean_value = init_weights[i, j]
            values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=shapes[ct])
            rep_weights[i * num_E:(i + 1) * num_E, j * num_E:(j + 1) * num_E] = values
            ct += 1

    plot = False
    if plot:
        plt.hist(rep_weights[:num_E, :num_E].flatten(), bins=50, alpha=.5, label='wee')
        plt.hist(rep_weights[:num_E, num_E:].flatten(), bins=50, alpha=.5, label='wei')
        plt.hist(rep_weights[num_E:, num_E:].flatten(), bins=50, alpha=.5, label='wii')
        plt.hist(rep_weights[num_E:, :num_E].flatten(), bins=50, alpha=.5, label='wie')
        plt.legend()
        plt.show()
    return rep_weights


def generate_block_wx(init_weights, N, N_X):
    """Generate the block-structured external weight matrix used in the simulation."""
    variances = [.000001, .000001]
    repeated_weights = np.zeros(shape=(N, N_X))
    shapes = N // 2
    ct = 0

    for i in range(init_weights.shape[0]):
        variance = variances[i]
        mean_value = init_weights[i]
        values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=(shapes, shapes))
        repeated_weights[i * N // 2:(i + 1) * N // 2, i * N // 2:(i + 1) * N // 2] = values

    plot = False
    if plot:
        plt.hist(repeated_weights.flatten(), alpha=.5, bins=1000)
        plt.show()
    return repeated_weights

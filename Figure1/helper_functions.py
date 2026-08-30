# Numerical, plotting, profiling, and file-system dependencies
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import time as TimeTime
import cProfile
import pstats
import pandas as pd
import os


# =============================================================================
# Ornstein-Uhlenbeck covariance
# =============================================================================

def OU_covariance_matrix(Sim_N, N_x, a, b, sigma_s, sigma_p, tau_ou, c):
    # Combine the shared covariance with population-specific private covariance.
    cov_matrix = (c * sigma_s**2) * np.ones((N_x, N_x))
    extra_diagonal = (1 - c) * sigma_p**2

    cov_matrix[:N_x//2, :N_x//2] += extra_diagonal
    cov_matrix[N_x//2:, N_x//2:] += extra_diagonal

    # Apply the OU frequency-domain filter and add finite-size diagonal terms.
    cov_matrix = cov_matrix * (1 / (1 + (2 * np.pi * tau_ou * 1j)**2))
    cov_matrix[np.arange(N_x//2), np.arange(N_x//2)] += a / Sim_N
    cov_matrix[N_x//2 + np.arange(N_x//2), N_x//2 + np.arange(N_x//2)] += b / Sim_N
    cov_matrix = Sim_N * cov_matrix * Sim_N

    return cov_matrix


# =============================================================================
# Spike-count statistics
# =============================================================================

def calculate_spike_count_covariance(spike_trains1, spike_trains2, window_size_ms, dt):
    """
    Calculate the average spike count covariance between neurons.

    Parameters:
    spike_trains (np.ndarray): An N x T binary array of spike trains (N neurons, T time points).
    window_size_ms (int): The size of the time window in milliseconds.
    tau_ou (float): The time step in milliseconds.

    Returns:
    np.ndarray: An N x N array of spike count covariances.
    """
    N_1, T = spike_trains1.shape
    N_2, T = spike_trains2.shape

    # Use the final quarter of each recording and divide it into counting windows.
    window_size = int(window_size_ms / dt)
    transient = 3 * T // 4
    num_windows = (T - transient) // window_size

    # Count spikes from each population in every window.
    print('num_windows', num_windows)
    spike_counts1 = np.zeros((N_1, num_windows))
    spike_counts2 = np.zeros((N_2, num_windows))

    for k in range(num_windows):
        binned_trains1 = bin_spike_trains(spike_trains1[:, transient:], dt, T)
        binned_trains2 = bin_spike_trains(spike_trains2[:, transient:], dt, T)
        start_idx = k * window_size
        end_idx = start_idx + window_size
        spike_counts1[:, k] = np.sum(binned_trains1[:, start_idx:end_idx], axis=1)
        spike_counts2[:, k] = np.sum(binned_trains2[:, start_idx:end_idx], axis=1)

    # Compute the covariance between every pair of neurons across windows.
    mean_spike_counts1 = np.mean(spike_counts1, axis=1)
    mean_spike_counts2 = np.mean(spike_counts2, axis=1)
    covariance_matrix = np.zeros((N_1, N_2))

    for i in range(N_1):
        for j in range(N_2):
            covariance_matrix[i, j] = np.mean((spike_counts1[i, :] - mean_spike_counts1[i]) * (spike_counts2[j, :] - mean_spike_counts2[j]))

    print('covariance_matrix', covariance_matrix)
    return covariance_matrix


# =============================================================================
# External-input generation
# =============================================================================

def generate_sep_poisson_neurons(base_ex, N_X, time_steps, dt, sigma_s, sigma_p, c_x):
    # Draw shared and private white-noise samples in the original order.
    xi_s = np.random.normal(0, 1, size=len(time_steps))
    xi_p = np.random.normal(0, 1, size=(N_X, len(time_steps)))

    # Construct the external intensity used by the current simulation.
    lambda_xx = base_ex + c_x * sigma_s * xi_s
    return lambda_xx


# =============================================================================
# Recurrent and external weight initialization
# =============================================================================

def generate_weight_array(init_weights, num_E, num_I):
    # Assign a small variance to each E/I connectivity block.
    variances = [.0000000001, .0000000001, .0000000001, .0000000001]

    row_num = [num_E, num_I]
    Num_neur = num_E + num_I
    rep_weights = np.zeros(shape=((num_E + num_I), (num_E + num_I)))
    shapes = [(num_E, num_E), (num_E, num_I), (num_I, num_E), (num_I, num_I)]
    ct = 0

    # Draw each population-to-population block from its original Gaussian.
    for i in range(init_weights.shape[0]):
        for j in range(init_weights.shape[1]):
            variance = variances[ct]
            mean_value = init_weights[i, j]
            values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=shapes[ct])
            rep_weights[i * num_E:(i + 1) * num_E, j * num_E:(j + 1) * num_E] = values
            ct += 1

    # Apply the original random sparsification and restore the two tracked entries.
    for j in range(int(num_E)):
        idxx = np.random.choice(np.arange(num_E), num_E)
        idxy = np.random.choice(np.arange(num_E), num_E)
        rep_weights[idxx, idxy] = 0
    rep_weights[5, 0] = mean_value
    rep_weights[4, 0] = mean_value

    return rep_weights


def generate_diagonal_wx(init_weights, N, N_X):
    # Build the original diagonal external-input matrix.
    variances = [.0001, .0001]
    repeated_weights = np.zeros(shape=(N, N_X))
    shapes = N // 2
    ct = 0

    for i in range(init_weights.shape[0]):
        variance = variances[i]
        mean_value = init_weights[i]
        values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=shapes)
        repeated_weights[np.arange(N//2), np.arange(N//2) + N//2] = values

    # Retain the original disabled diagnostic plot.
    plot = False
    if plot:
        plt.hist(repeated_weights.flatten(), alpha=.5)
        plt.show()

    return repeated_weights


def generate_block_wx(init_weights, N, N_X):
    # Build separate external-input blocks for the two network populations.
    variances = [.000000001, .000000001]
    repeated_weights = np.zeros(shape=(N, N_X))
    shapes = N // 2
    ct = 0

    for i in range(init_weights.shape[0]):
        variance = variances[i]
        mean_value = init_weights[i]
        values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=(shapes, shapes))
        repeated_weights[i * N//2:(i + 1) * N//2, i * N//2:(i + 1) * N//2] = values

    return repeated_weights


# =============================================================================
# Windowed firing-rate calculation
# =============================================================================

def calculate_firing_rate(N, spike_times, window_size, time_step, curr_t):
    firing_rate = np.zeros(N)

    # Count each neuron's spikes in the requested window and convert to a rate.
    for i in range(N):
        start = int(curr_t - window_size)
        end = int(curr_t)
        spike_count = np.count_nonzero(spike_times[i, start:end])
        firing_rate[i] = spike_count / (window_size * time_step)

    return firing_rate

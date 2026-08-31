import numpy as np
import matplotlib.pyplot as plt
# from HelperMFTfunctions import *
import ctypes
import time as TimeTime
import cProfile
import pstats
import pandas as pd
import os
from helper_functions import *

save = True

# import helper
# wies = np.linspace(2., 3.5, 3)
# wiis = [30.]  # np.linspace(3., 4., 2)


def spiking_sim(normalize, axs, seed, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II):
    # Report the current simulation configuration.
    print('c_X', c_x, 'seed', seed, 'taur', tau_r, 'normalize', normalize)

    # Define network, simulation, and plasticity parameters.
    ms_per_sec = 1000
    Num_neur = N_E + N_I
    ctt = 0
    ms_per_sec = 1000
    Num_neur = N_E + N_I
    burnin = 0.
    time = np.linspace(0, T, int(T / dt))
    B12 = 1 / 2
    B21 = 1 / 2

    b = 20.
    B12_inh = 1 / 2
    B21_inh = 1 / 2

    alpha_ee = b * 2 * tau_STDP
    alpha_ei = b * 2 * tau_STDP

    start_time = TimeTime.time()

    np.random.seed(988 * seed)
    plastic = True
    eta_ee = 1 / tau_wee
    eta_ei = 1 / tau_wei
    sigma_s = sigma
    sigma_p = sigma
    num_per_pool = 1
    N_x = Num_neur * num_per_pool
    print('Nx', N_x)
    base_ex = 20

    # Initialize spike-history and synaptic-filter traces.
    x = np.zeros(shape=(1, Num_neur))
    g_elig_E = np.zeros(shape=(N_E))
    g_elig_I = np.zeros(shape=(N_I))
    g_elig_X = np.zeros(shape=(N_x))

    # Initialize recurrent E-I weights.
    W = np.zeros(shape=(Num_neur, Num_neur))
    init_weights = np.array([[w_EE, w_EI], [w_IE, w_II]])

    # Initialize external-input weights.
    # Px = np.array([[1.], [1.]])
    Wxm = np.array([.04, .03]) / np.sqrt(N_E)
    Wx_theory = np.array([[Wxm[0], 0], [0, Wxm[1]]])

    bernouilli = False
    if bernouilli:
        W_mat_init = generate_bernouilli_weight_array(init_weights / .1, N_E, N_I)
        Wx, p = generate_bernouilli_wx(Wxm / .1, Num_neur, N_x, num_per_pool)
    else:
        Wx = generate_block_wx(Wxm, Num_neur, N_x)
        W_mat_init = generate_weight_array(init_weights, N_E, N_I)

    # Apply inhibitory signs and enforce the intended block structure.
    W_mat_init[:, N_E:] = W_mat_init[:, N_E:] * -1
    W[:, :] = W_mat_init

    idx_ie = np.where(W[N_E:, :N_E] < 0)
    idx_ii = np.where(W[N_E:, N_E:] > 0)
    W[N_E + idx_ie[0], idx_ie[1]] = 0
    W[N_E + idx_ii[0], N_E + idx_ii[1]] = 0

    # Configure data storage and select recurrent weights to track.
    store_rates = True
    store_weights = True
    store_mean_weights = True
    store_theory_lamb = True
    test_theory = False
    track_num_n = 1
    skip = 1000
    track_mean_W = np.zeros(shape=(2, int(len(time) // skip) + 1))

    if store_weights:
        track_W = np.zeros(shape=(2 * track_num_n, int(len(time) // skip) + 1))

        neur_idx = np.arange(N_E * N_E)
        random_weights_to_store = np.random.choice(neur_idx, track_num_n)
        x_idx_e = random_weights_to_store // N_E
        y_idx_e = random_weights_to_store % N_E

        neur_idx = np.arange(N_E * N_I)
        random_weights_to_store = np.random.choice(neur_idx, track_num_n)
        x_idx_i = random_weights_to_store // N_E
        y_idx_i = random_weights_to_store % N_I
    else:
        track_W = np.array([0, 0, 0])

    # Allocate spike-train storage.
    bins_store = int(T / dt)
    if store_rates:
        spike_trains_E = np.zeros((N_E, bins_store))
        spike_trains_I = np.zeros((N_I, bins_store))
        spike_trains_X = np.zeros((N_x, bins_store))
        print('CHECK HERE', N_E, bins_store)
    else:
        spike_trains_E = 0
        spike_trains_I = 0
        spike_trains_X = 0

    # Allocate reduced-resolution tracking arrays.
    # track_x = np.zeros(shape=(Num_neur, int(len(time) // skip)))
    track_mean_x = np.zeros(shape=(2, int(len(time) // skip) + 1))
    # track_std_x = np.zeros(shape=(2, int(len(time) // skip)))
    track_lam_E = np.zeros(shape=(track_num_n, int(len(time) // skip) + 1))
    track_lam_I = np.zeros(shape=10)
    # track_rates = np.zeros(shape=(Num_neur, len(time)))
    track_mean_rates = np.zeros(shape=(2, int(len(time) // skip) + 1))

    tr = np.array([0, 0])
    tw = np.array([w_EE, -w_EI])
    if bernouilli:
        tw = tw / .1

    balance_index = None
    # balance_index = np.zeros(int(len(time) // skip) + 1)

    # Initialize external rates, filters, and OU processes.
    aE = base_ex
    aI = base_ex

    ex_weights = Wx
    ex_weights_E = ex_weights[:N_E, :N_x // 2]
    ex_weights_I = ex_weights[N_E:, N_x // 2:]

    lambda_X = np.zeros(shape=(N_x))
    lambda_xe = np.zeros(shape=(N_x // 2))
    lambda_xi = np.zeros(shape=(N_x // 2))
    x_ou_s = 0
    x_ou_pe = np.zeros(shape=(N_x // 2))
    x_ou_pi = np.zeros(shape=(N_x // 2))

    np.random.seed(9 * seed)
    np.fill_diagonal(W, 0)
    track_lamx = []
    track_gex = []

    max_events_to_store = 10000
    event_count = 0
    event_count_i = 0

    # Advance the network, external OU drive, and plastic synapses.
    for t in range(len(time) - 1):
        np.fill_diagonal(W, 0)
        if bernouilli:
            W[w_idx_zeros] = 0

        # Construct excitatory and inhibitory external-input intensities.
        if time[t] > 0.:
            lambda_xe = aE + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pe
            lambda_xi = aI + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pi
        else:
            lambda_xe = 0
            lambda_xi = 0

        lambda_X[:N_x // 2] = lambda_xe
        lambda_X[N_x // 2:] = lambda_xi
        if t % skip == 0:
            track_lamx.append(lambda_X[0])

        # Update the shared and private OU processes.
        dx_s = -x_ou_s * dt / tau_ou + (sigma_s / tau_ou) * np.random.normal(0, 1) * np.sqrt(dt)
        x_ou_s += dx_s

        dx_pe = -x_ou_pe * dt / tau_ou + (sigma_p / tau_ou) * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe

        dx_pi = -x_ou_pi * dt / tau_ou + (sigma_p / tau_ou) * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pi += dx_pi

        # Store selected recurrent weights.
        if store_weights and t % skip == 0:
            track_W[:track_num_n, t // skip] = W[x_idx_e, y_idx_e]
            track_W[track_num_n:, t // skip] = W[x_idx_i, N_E + y_idx_i]

        # Calculate external, recurrent, and total intensities.
        ext_act = ex_weights @ g_elig_X
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act + ext_act

        idx = np.where(Lam < 0)
        Lam[idx] = 0

        # Draw spikes for the E, I, and external populations.
        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)

        if t < bins_store:
            if store_rates:
                spike_trains_E[random_numE < (Lam[:N_E] * dt), t] = time[t]
                spike_trains_I[random_numI < (Lam[N_E:] * dt), t] = time[t]
                spike_trains_X[random_numX < (lambda_X * dt), t] = time[t]
            idx1 = t
        else:
            if store_rates:
                spike_trains_E[:, :-1] = spike_trains_E[:, 1:]
                spike_trains_I[:, :-1] = spike_trains_I[:, 1:]
                spike_trains_X[:, :-1] = spike_trains_X[:, 1:]
                spike_trains_E[random_numE > (Lam[:N_E] * dt), -1] = 0
                spike_trains_I[random_numI > (Lam[N_E:] * dt), -1] = 0
                spike_trains_X[random_numX > (lambda_X * dt), -1] = 0

                spike_trains_E[random_numE < (Lam[:N_E] * dt), -1] = time[t]
                spike_trains_I[random_numI < (Lam[N_E:] * dt), -1] = time[t]
                spike_trains_X[random_numX < (lambda_X * dt), -1] = time[t]
            idx1 = bins_store - 1

        Espike = np.argwhere(random_numE < (Lam[:N_E] * dt))
        Ispike = np.argwhere(random_numI < (Lam[N_E:] * dt))

        # Store reduced-resolution intensities and mean traces.
        if t % skip == 0:
            track_gex.append(ext_act)
            track_mean_x[0, t // skip] = np.mean(x[0, :N_E])
            track_mean_x[1, t // skip] = np.mean(x[0, N_E:])

            track_lam_E[:track_num_n, t // skip] = Lam[:track_num_n]
            track_mean_rates[0, t // skip] = np.mean(Lam[:N_E])
            track_mean_rates[1, t // skip] = np.mean(Lam[N_E:])

        # Let the network approach steady state before enabling plasticity.
        start_plastic = 4.
        if plastic and time[t] > start_plastic:
            W[Espike[Espike <= N_E], :N_E] += np.tile(eta_ee * B12 * x[0, :N_E].transpose(), (np.count_nonzero(Espike <= N_E), 1))
            W[:N_E, Espike[Espike <= N_E]] += np.tile(eta_ee * B21 * (x[0, :N_E] - alpha_ee), (np.count_nonzero(Espike <= N_E), 1)).transpose()
            W[:N_E, Ispike[Ispike <= N_I] + N_E] -= np.tile(eta_ei * B21_inh * (x[0, :N_E] - alpha_ei), (np.count_nonzero(Ispike <= N_I), 1)).transpose()
            W[Espike[Espike <= N_E], N_E:] -= np.tile(eta_ei * B12_inh * x[0, N_E:].transpose(), (np.count_nonzero(Espike < N_E), 1))

        # Enforce zero self-connections and the E/I sign constraints.
        np.fill_diagonal(W, 0)
        W[:N_E, :N_E][W[:N_E, :N_E] < 0] = 0
        W[:N_E, N_E:][W[:N_E, N_E:] > 0] = 0

        # Decay recurrent and plasticity eligibility traces.
        tau_r_I = 2 * tau_r
        x[0, :N_E] -= dt * x[0, :N_E] / tau_STDP
        x[0, N_E:] -= dt * x[0, N_E:] / tau_STDP
        g_elig_E -= dt * g_elig_E / tau_r
        g_elig_I -= dt * g_elig_I / tau_r_I
        g_elig_X -= dt * g_elig_X / tau_r

        # Add excitatory spikes to the traces.
        idx = np.where(random_numE < (Lam[:N_E] * dt))
        x[0, idx] += 1
        if normalize:
            g_elig_E[idx] += 1 / tau_r
        else:
            g_elig_E[idx] += 1

        # Add inhibitory spikes to the traces.
        idx = np.where(random_numI < (Lam[N_E:] * dt))
        if normalize:
            g_elig_I[idx] += 1 / tau_r_I
        else:
            g_elig_I[idx] += 1
        idx = N_E + np.array(idx)
        x[0, idx] += 1

        # Add external spikes to the synaptic filter.
        idx = np.where(random_numX < (lambda_X * dt))
        if normalize:
            g_elig_X[idx] += 1 / tau_r
        else:
            g_elig_X[idx] += 1

        # Print periodic simulation diagnostics.
        if t % 10000 == 0:
            print('======================')
            print('======================')
            print('t', t, 'time[t]', time[t], 'idx1', idx1)
            print('ACTUAL LAMB', 'poisson labmb', np.mean(Lam[:N_E]), 'mean I rates', np.mean(Lam[N_E:]), 'track_rates[0,t]')

        # Store the mean nonzero E-to-E and I-to-E weights.
        if store_mean_weights and t % skip == 0:
            ee_weights = W[:N_E, :N_E]
            ei_weights = W[:N_E, N_E:]
            nonzero_elements_ee = ee_weights[ee_weights != 0]
            nonzero_elements_ei = ei_weights[ei_weights != 0]
            mean_nonzero_ee = np.mean(nonzero_elements_ee)
            mean_nonzero_ei = np.mean(nonzero_elements_ei)

            if bernouilli:
                track_mean_W[0, t // skip] = mean_nonzero_ee
                track_mean_W[1, t // skip] = mean_nonzero_ei
            else:
                track_mean_W[0, t // skip] = mean_nonzero_ee
                track_mean_W[1, t // skip] = mean_nonzero_ei

            if t % 20000 == 0:
                print('time', time[t])
                print('track_mean_W_ee[t]', mean_nonzero_ee)
                print('track_mean_W_ei[t]', mean_nonzero_ei)

    # Construct the reduced time axis after the simulation finishes.
    time_reduced = time[::skip]

    # Optionally plot diagnostic weight, intensity, and input traces.
    if axs is not None:
        axs[0].plot(time_reduced, track_W[:track_num_n, :-1].T, color='tab:grey', linewidth=.5)
        print('track_mean_W[:track_num_n,:-1]', track_mean_W[0, :10])
        axs[0].plot(time_reduced, track_mean_W[0, :-1], color='tab:red', linestyle='solid', label='sim ee')

        axs[0].plot(time_reduced, track_W[track_num_n:, :-1].T, color='tab:grey', linewidth=.5)
        axs[0].plot(time_reduced, track_mean_W[1, :-1], color='tab:blue', linestyle='solid', linewidth=1., label='sim ei')

        axs[1].plot(time_reduced, track_lam_E[:, :-1].T, color='tab:grey', linewidth=1.)
        axs[1].plot(time_reduced, track_mean_rates[0, :-1], color='red', label='mean ')
        axs[3].plot(time_reduced, track_gex, label='theory ex', linewidth=1.)
        # axs[1].axhline(b, color='k')
        axs[2].plot(time_reduced, track_lamx)
        axs[1].set_ylim(0, 35)
        axs[0].set_ylim(-.0025, .0025)
        axs[2].set_ylim(0, 55)
        # axs[0].set_ylim(-1.5 / 1000, 1. / 1000)

        for j in range(3):
            # axs[j].legend()
            axs[j].spines['top'].set_visible(False)
            axs[j].spines['right'].set_visible(False)
            axs[j].spines['bottom'].set_linewidth(1)

    # Package the tracked mean recurrent weights for downstream analysis.
    track_W_mean = np.vstack([track_mean_W[0, :], track_mean_W[1, :]])

    # return time, track_x, track_W, track_W_mean, track_W_std, track_rates, track_mean_x, track_std_x, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, W_ee_theory, W_ei_theory, theory_lamb, balance_index
    return time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, balance_index, b

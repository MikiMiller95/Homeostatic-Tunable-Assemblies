# Generate the spike-time data used in the asynchronous raster panel.

import numpy as np
import pandas as pd

from helper_functions import generate_block_wx, generate_weight_array


# Simulate the recurrent E-I network and return the stored spike and rate arrays.
def spiking_sim(seed, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II):
    # Set the simulation duration, network size, and homeostatic target.
    Num_neur = N_E + N_I
    time = np.linspace(0, T, int(T / dt))
    b = 20.0

    # Seed and initialize the network, external input, and eligibility traces.
    np.random.seed(988 * seed)
    sigma_s = sigma
    sigma_p = sigma
    num_per_pool = 1
    N_x = Num_neur * num_per_pool
    base_ex = 20
    x = np.zeros(shape=(1, Num_neur))
    g_elig_E = np.zeros(shape=N_E)
    g_elig_I = np.zeros(shape=N_I)
    g_elig_X = np.zeros(shape=N_x)

    # Construct the recurrent and external connectivity matrices.
    W = np.zeros(shape=(Num_neur, Num_neur))
    init_weights = np.array([[w_EE, w_EI], [w_IE, w_II]])
    Wxm = np.array([0.04, 0.03]) / np.sqrt(N_E)
    Wx = generate_block_wx(Wxm, Num_neur, N_x)
    W_mat_init = generate_weight_array(init_weights, N_E, N_I)
    W_mat_init[:, N_E:] = W_mat_init[:, N_E:] * -1
    W[:, :] = W_mat_init
    idx_ie = np.where(W[N_E:, :N_E] < 0)
    idx_ii = np.where(W[N_E:, N_E:] > 0)
    W[N_E + idx_ie[0], idx_ie[1]] = 0
    W[N_E + idx_ii[0], N_E + idx_ii[1]] = 0

    # Allocate arrays for stored weights, intensities, rates, and spike times.
    store_weights = True
    track_num_n = N_E
    skip = 1
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
    bins_store = int(T / dt)
    spike_trains_E = np.zeros((N_E, bins_store))
    spike_trains_I = np.zeros((N_I, bins_store))
    spike_trains_X = np.zeros((N_x, bins_store))
    track_lam_E = np.zeros(shape=(track_num_n, int(len(time) // skip) + 1))
    track_lam_I = np.zeros(shape=(track_num_n, int(len(time) // skip) + 1))


    track_rates = np.zeros(shape=(10, len(time)))
    track_mean_rates = np.zeros(shape=(2, int(len(time) // skip) + 1))
    aE = base_ex
    aI = base_ex
    ex_weights = Wx

    # Initialize the shared and population-specific Ornstein-Uhlenbeck inputs.
    lambda_X = np.zeros(shape=N_x)
    x_ou_s = 0
    x_ou_pe = np.zeros(shape=N_x // 2)
    x_ou_pi = np.zeros(shape=N_x // 2)
    np.random.seed(94 * seed)
    np.fill_diagonal(W, 0)

    # Advance the network one time step at a time.
    for t in range(len(time) - 1):
        np.fill_diagonal(W, 0)
        # Turn on the external drive after the initial 100 ms.
        if time[t] > 0.1:
            lambda_xe = aE + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pe
            lambda_xi = aI + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pi
        else:
            lambda_xe = 0
            lambda_xi = 0
        lambda_X[:N_x // 2] = lambda_xe
        lambda_X[N_x // 2:] = lambda_xi

        # Update the shared and population-specific OU processes.
        dx_s = -x_ou_s * dt / tau_ou + sigma_s / tau_ou * np.random.normal(0, 1) * np.sqrt(dt)
        x_ou_s += dx_s
        dx_pe = -x_ou_pe * dt / tau_ou + sigma_p / tau_ou * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe
        dx_pi = -x_ou_pi * dt / tau_ou + sigma_p / tau_ou * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pi += dx_pi
        if store_weights and t % skip == 0:
            track_W[:track_num_n, t // skip] = W[x_idx_e, y_idx_e]
            track_W[track_num_n:, t // skip] = W[x_idx_i, N_E + y_idx_i]

        # Compute the recurrent and external contributions to the intensity.
        ext_act = ex_weights @ g_elig_X
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act + ext_act
        idx = np.where(Lam < 0)
        Lam[idx] = 0

        # Draw excitatory, inhibitory, and external spikes from the intensities.
        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)
        if t < bins_store:
            spike_trains_E[random_numE < Lam[:N_E] * dt, t] = time[t]
            spike_trains_I[random_numI < Lam[N_E:] * dt, t] = time[t]
            spike_trains_X[random_numX < lambda_X * dt, t] = time[t]
        else:
            spike_trains_E[:, :-1] = spike_trains_E[:, 1:]
            spike_trains_I[:, :-1] = spike_trains_I[:, 1:]
            spike_trains_X[:, :-1] = spike_trains_X[:, 1:]
            spike_trains_E[random_numE > Lam[:N_E] * dt, -1] = 0
            spike_trains_I[random_numI > Lam[N_E:] * dt, -1] = 0
            spike_trains_X[random_numX > lambda_X * dt, -1] = 0
            spike_trains_E[random_numE < Lam[:N_E] * dt, -1] = time[t]
            spike_trains_I[random_numI < Lam[N_E:] * dt, -1] = time[t]
            spike_trains_X[random_numX < lambda_X * dt, -1] = time[t]

        # Store the current rates and selected network quantities.
        if t % skip == 0:
            track_lam_E[:track_num_n, t // skip] = Lam[:track_num_n]
            track_lam_I[:track_num_n, t // skip] = Lam[N_E:N_E + track_num_n]
            track_mean_rates[0, t // skip] = np.mean(Lam[:N_E])
            track_mean_rates[1, t // skip] = np.mean(Lam[N_E:])
        np.fill_diagonal(W, 0)

        # Decay the STDP and synaptic traces, then increment them at spike times.
        tau_r_I = 2 * tau_r
        x[0, :N_E] -= dt * x[0, :N_E] / tau_STDP
        x[0, N_E:] -= dt * x[0, N_E:] / tau_STDP
        g_elig_E -= dt * g_elig_E / tau_r
        g_elig_I -= dt * g_elig_I / tau_r_I
        g_elig_X -= dt * g_elig_X / tau_r
        idx = np.where(random_numE < Lam[:N_E] * dt)
        x[0, idx] += 1
        g_elig_E[idx] += 1 / tau_r
        idx = np.where(random_numI < Lam[N_E:] * dt)
        g_elig_I[idx] += 1 / tau_r_I
        idx = N_E + np.array(idx)
        x[0, idx] += 1
        idx = np.where(random_numX < lambda_X * dt)
        g_elig_X[idx] += 1 / tau_r

    # Package the recorded arrays for the caller.
    time_reduced = time[::skip]
    track_W_mean = np.vstack([track_mean_W[0, :], track_mean_W[1, :]])
    return (time, track_W, track_W_mean, track_rates, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, spike_trains_X, b)

import gc

cases = [
    {
        'name': 'current weak',
        'N_E': 500,
        'N_I': 500,
        'T': 10.0,
        'dt': 0.1 / 1000.0,
        'tau_ou': 8.0 / 1000.0,
        'tau_r': 8.0 / 1000.0,
        'tau_STDP': 40.0 / 1000.0,
        'tau_wei': 75000 / 3,
        'tau_wee': 225000 / 3,
        'w_EE_scale': 0.036,
        'w_EI_scale': 0.036,
        'w_IE_scale': 0.02,
        'w_II_scale': 0.02,
        'seed': 0
    },
]

sigma = 0.0
c_x = 0.0
ctt = 0
loc = 'Fig1Rastordata/'

for case in cases:
    N_E = case['N_E']
    N_I = case['N_I']
    T = case['T']
    dt = case['dt']
    tau_ou = case['tau_ou']
    tau_r = case['tau_r']
    tau_STDP = case['tau_STDP']
    tau_wei = case['tau_wei']
    tau_wee = case['tau_wee']
    seed = case['seed']

    factor = np.sqrt(N_E)
    w_EE = case['w_EE_scale'] / factor
    w_EI = case['w_EI_scale'] / factor
    w_IE = case['w_IE_scale'] / factor
    w_II = case['w_II_scale'] / factor

    print(f"Running {case['name']} case")

    time, track_W, track_W_mean, track_rates, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, spike_trains_X, cval = spiking_sim(seed, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II)

    lam_suffix = f'Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}'
    spike_suffix = f'Ne_{N_E}_T{T}_wee{w_EE * factor}_wei{w_EI * factor}_wii{w_II * factor}_wie{w_IE * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}'

    df_lamE = pd.DataFrame(track_lam_E)
    df_lamE.to_csv(f'{loc}tracked_lamE_{lam_suffix}.csv', index=False)

    df_lamI = pd.DataFrame(track_lam_I)
    df_lamI.to_csv(f'{loc}tracked_lamI_{lam_suffix}.csv', index=False)

    df_spikes_E = pd.DataFrame(spike_trains_E)
    df_spikes_E.to_csv(f'{loc}spike_trains_E_{spike_suffix}.csv', index=False)

    df_spikes_I = pd.DataFrame(spike_trains_I)
    df_spikes_I.to_csv(f'{loc}spike_trains_I_{spike_suffix}.csv', index=False)

    df_spikes_X = pd.DataFrame(spike_trains_X)
    df_spikes_X.to_csv(f'{loc}spike_trains_X_{spike_suffix}.csv', index=False)

    print(f"Finished {case['name']} case")

    del time, track_W, track_W_mean, track_rates, track_mean_rates, time_reduced
    del track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, spike_trains_X
    del df_lamE, df_lamI, df_spikes_E, df_spikes_I, df_spikes_X
    gc.collect()

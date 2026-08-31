import numpy as np

from helper_functions import *


def spiking_sim(normalize, axs, seed, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II, pulse_num=0):
    """Run the spiking E/I network and return the quantities saved for Figure 2."""
    print('c_X', c_x, 'seed', seed, 'taur', tau_r, 'normalize', normalize)

    # Network, plasticity-rule, and time-grid settings.
    ms_per_sec = 1000
    Num_neur = N_E + N_I
    time = np.linspace(0, T, int(T / dt))
    B12 = 1 / 2
    B21 = 1 / 2
    b = 20.
    B12_inh = 1 / 2
    B21_inh = 1 / 2
    alpha_ee = b * 2 * tau_STDP
    alpha_ei = b * 2 * tau_STDP

    np.random.seed(98 * seed)
    plastic = True
    eta_ee = 1 / tau_wee
    eta_ei = 1 / tau_wei
    sigma_s = sigma
    sigma_p = sigma
    num_per_pool = 1
    N_x = Num_neur * num_per_pool
    print('Nx', N_x)
    base_ex = 20

    # Initialize synaptic and eligibility traces.
    x = np.zeros(shape=(1, Num_neur))
    g_elig_E = np.zeros(shape=N_E)
    g_elig_I = np.zeros(shape=N_I)
    g_elig_X = np.zeros(shape=N_x)

    # Initialize recurrent and external connectivity.
    W = np.zeros(shape=(Num_neur, Num_neur))
    init_weights = np.array([[w_EE, w_EI], [w_IE, w_II]])
    Wxm = np.array([.04, .03]) / np.sqrt(N_E)

    bernouilli = False
    if bernouilli:
        W_mat_init = generate_bernouilli_weight_array(init_weights / .1, N_E, N_I)
        Wx, p = generate_bernouilli_wx(Wxm / .1, Num_neur, N_x, num_per_pool)
        w_idx_zeros = np.where(W_mat_init == 0.)
    else:
        Wx = generate_block_wx(Wxm, Num_neur, N_x)
        W_mat_init = generate_weight_array(init_weights, N_E, N_I)

    W_mat_init[:, N_E:] = W_mat_init[:, N_E:] * -1
    W[:, :] = W_mat_init

    # Enforce the intended signs of the fixed E-to-I and I-to-I blocks.
    idx_ie = np.where(W[N_E:, :N_E] < 0)
    idx_ii = np.where(W[N_E:, N_E:] > 0)
    W[N_E + idx_ie[0], idx_ie[1]] = 0
    W[N_E + idx_ii[0], N_E + idx_ii[1]] = 0

    # Configure the quantities retained at the reduced sampling interval.
    store_weights = True
    store_mean_weights = True
    track_num_n = 1
    skip = 50

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

    # Optional perturbations are retained but are not enabled by RunFig2Spiking.py.
    if pulse_num == 1:
        pert_time = 6.8
        pulse_amp = 20.
    elif pulse_num > 1:
        pulse_amp = 50.
        pulse_width = .02
        pulse_interval = 2.
        pulse_start = 15.0
        pulse_times = np.arange(pulse_start, T, pulse_interval)
    else:
        pulse_amp = 0.

    track_lam_E = np.zeros(shape=(track_num_n, int(len(time) // skip) + 1))
    track_lam_I = np.zeros(shape=10)
    track_mean_rates = np.zeros(shape=(2, int(len(time) // skip) + 1))
    balance_index = None

    # External-input rates and OU state variables.
    aE = base_ex
    aI = base_ex
    ex_weights = Wx

    lambda_X = np.zeros(shape=N_x)
    lambda_xe = np.zeros(shape=N_x // 2)
    lambda_xi = np.zeros(shape=N_x // 2)
    x_ou_s = 0
    x_ou_pe = np.zeros(shape=N_x // 2)
    x_ou_pi = np.zeros(shape=N_x // 2)

    np.random.seed(9 * seed)
    np.fill_diagonal(W, 0)

    # Integrate the external drive, spikes, eligibility traces, and plastic weights.
    for t in range(len(time) - 1):
        np.fill_diagonal(W, 0)

        if bernouilli:
            W[w_idx_zeros] = 0

        if time[t] > 0.:
            lambda_xe = aE + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pe
            lambda_xi = aI + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pi
        else:
            lambda_xe = 0
            lambda_xi = 0

        lambda_X[:N_x // 2] = lambda_xe
        lambda_X[N_x // 2:] = lambda_xi

        dx_s = -x_ou_s * dt / tau_ou + (sigma_s / tau_ou) * np.random.normal(0, 1) * np.sqrt(dt)
        x_ou_s += dx_s

        dx_pe = -x_ou_pe * dt / tau_ou + (sigma_p / tau_ou) * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe

        dx_pi = -x_ou_pi * dt / tau_ou + (sigma_p / tau_ou) * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pi += dx_pi

        if store_weights and t % skip == 0:
            track_W[:track_num_n, t // skip] = W[x_idx_e, y_idx_e]
            track_W[track_num_n:, t // skip] = W[x_idx_i, N_E + y_idx_i]

        # Convert eligibility traces into recurrent and external intensities.
        ext_act = ex_weights @ g_elig_X
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act + ext_act

        if pulse_num == 1:
            if abs(time[t] - pert_time) < 20 * dt:
                Lam += pulse_amp
        elif pulse_num > 1:
            if np.any(np.abs(time[t] - pulse_times) < pulse_width / 2):
                Lam[:N_E] += pulse_amp

        Lam[np.where(Lam < 0)] = 0

        # Draw E, I, and external spikes from the current intensities.
        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)
        Espike = np.argwhere(random_numE < Lam[:N_E] * dt)
        Ispike = np.argwhere(random_numI < Lam[N_E:] * dt)

        if t % skip == 0:
            track_lam_E[:track_num_n, t // skip] = Lam[:track_num_n]
            track_mean_rates[0, t // skip] = np.mean(Lam[:N_E])
            track_mean_rates[1, t // skip] = np.mean(Lam[N_E:])

        # Allow four seconds of rate relaxation before enabling plasticity.
        start_plastic = 4.
        if plastic and time[t] > start_plastic:
            W[Espike[Espike <= N_E], :N_E] += np.tile(eta_ee * B12 * x[0, :N_E].transpose(), (np.count_nonzero(Espike <= N_E), 1))
            W[:N_E, Espike[Espike <= N_E]] += np.tile(eta_ee * B21 * (x[0, :N_E] - alpha_ee), (np.count_nonzero(Espike <= N_E), 1)).transpose()
            W[:N_E, Ispike[Ispike <= N_I] + N_E] -= np.tile(eta_ei * B21_inh * (x[0, :N_E] - alpha_ei), (np.count_nonzero(Ispike <= N_I), 1)).transpose()
            W[Espike[Espike <= N_E], N_E:] -= np.tile(eta_ei * B12_inh * x[0, N_E:].transpose(), (np.count_nonzero(Espike < N_E), 1))

        # Reapply sign constraints and decay all eligibility traces.
        np.fill_diagonal(W, 0)
        W[:N_E, :N_E][W[:N_E, :N_E] < 0] = 0
        W[:N_E, N_E:][W[:N_E, N_E:] > 0] = 0

        tau_r_I = 2 * tau_r
        x[0, :N_E] -= dt * x[0, :N_E] / tau_STDP
        x[0, N_E:] -= dt * x[0, N_E:] / tau_STDP
        g_elig_E -= dt * g_elig_E / tau_r
        g_elig_I -= dt * g_elig_I / tau_r_I
        g_elig_X -= dt * g_elig_X / tau_r

        # Add the new E spikes to the STDP and synaptic traces.
        idx = np.where(random_numE < Lam[:N_E] * dt)
        x[0, idx] += 1
        if normalize:
            g_elig_E[idx] += 1 / tau_r
        else:
            g_elig_E[idx] += 1

        # Add the new I spikes to the STDP and synaptic traces.
        idx = np.where(random_numI < Lam[N_E:] * dt)
        if normalize:
            g_elig_I[idx] += 1 / tau_r_I
        else:
            g_elig_I[idx] += 1
        idx = N_E + np.array(idx)
        x[0, idx] += 1

        # Add the new external spikes to their synaptic traces.
        idx = np.where(random_numX < lambda_X * dt)
        if normalize:
            g_elig_X[idx] += 1 / tau_r
        else:
            g_elig_X[idx] += 1

        # Record population-mean nonzero E-to-E and I-to-E weights.
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

    time_reduced = time[::skip]
    print('reduced time', np.shape(time_reduced))

    track_W_mean = np.vstack([track_mean_W[0, :], track_mean_W[1, :]])
    return time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, balance_index, b

"""Spiking-network simulation used for the third row of Figure 2."""

import time as TimeTime

import numpy as np

from helper_functions import generate_bernouilli_weight_array, generate_bernouilli_wx, generate_block_wx, generate_weight_array


def spiking_sim(normalize, axs, seed, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II, pulse_num=0, pattern_comp=False):
    """Run the spiking simulation without changing the original update rules."""
    print('c_X', c_x, 'seed', seed, 'taur', tau_r, 'normalize', normalize)

    # Shared constants and plasticity parameters.
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
    g_elig_E = np.zeros(shape=N_E)
    g_elig_I = np.zeros(shape=N_I)
    g_elig_X = np.zeros(shape=N_x)
    window_size = int((300 / ms_per_sec) / dt)

    # Construct recurrent and external weight matrices.
    print('make W')
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
    print('after make W')

    W_mat_init[:, N_E:] = W_mat_init[:, N_E:] * -1
    W[:, :] = W_mat_init

    idx_ie = np.where(W[N_E:, :N_E] < 0)
    idx_ii = np.where(W[N_E:, N_E:] > 0)
    W[N_E + idx_ie[0], idx_ie[1]] = 0
    W[N_E + idx_ii[0], N_E + idx_ii[1]] = 0

    # Configure the perturbation protocol. Figure 2 row 3 uses pulse_num > 1.
    store_rates = True
    store_weights = True
    store_mean_weights = True
    track_num_n = 1

    if pulse_num == 1:
        skip = 1
        pert_time = 6.8
        pulse_amp = .02 / dt
        print('INSIDE THE PULSE 1')
    elif pulse_num > 1:
        skip = 100
        pulse_amp = .015
        pulse_width = 0.03
        burst_duration = 100.0
        silent_duration = 0.0
        pulse_interval = 5.
        pulse_start = 20.
        cycle_duration = burst_duration + silent_duration

        pulse_times = []
        t = pulse_start
        while t < T:
            burst_end = t + burst_duration
            burst_pulses = np.arange(t, min(burst_end, T), pulse_interval)
            pulse_times.extend(burst_pulses)
            t += cycle_duration

        width_steps = int(np.round(pulse_width / dt))
        burst_mask = np.zeros_like(time, dtype=bool)
        for pt in pulse_times:
            i0 = int(np.round(pt / dt))
            burst_mask[i0:i0 + width_steps] = True
    else:
        skip = 1
        pulse_amp = 0.

    print('skip ', skip)

    # Allocate the recorded quantities written by the runner.
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

    track_lam_E = np.zeros(shape=(track_num_n, int(len(time) // skip) + 1))
    track_lam_I = np.zeros(shape=10)
    track_mean_rates = np.zeros(shape=(2, int(len(time) // skip) + 1))
    track_var_rates = np.zeros(shape=(1, int(len(time) // skip) + 1))

    # Initialize external input and OU variables.
    aE = base_ex
    aI = base_ex
    ex_weights = Wx
    ex_weights_E = ex_weights[:N_E, :N_x // 2]
    ex_weights_I = ex_weights[N_E:, N_x // 2:]

    lambda_X = np.zeros(shape=N_x)
    lambda_xe = np.zeros(shape=N_x // 2)
    lambda_xi = np.zeros(shape=N_x // 2)
    x_ou_s = 0
    x_ou_pe = np.zeros(shape=N_x // 2)
    x_ou_pi = np.zeros(shape=N_x // 2)
    np.random.seed(9 * seed)
    np.fill_diagonal(W, 0)
    track_lamx = []
    track_gex = []

    max_events_to_store = 10000
    event_count = 0
    event_count_i = 0
    delta_t_list = []
    delta_w_list = []
    i2e_delta_t_list = []
    i2e_delta_w_list = []

    print('outside loops')
    Lam = [0]

    # Retain the original optional pattern-completion branch used by other calls.
    if pattern_comp:
        pat_frac = .20
        pat_size = max(1, int(pat_frac * N_E))
        rng_pat = np.random.default_rng(12345 + int(seed))
        pattern_idx = rng_pat.choice(np.arange(N_E), size=pat_size, replace=False)
        pat_time = 5.0
        pat_amp = 0.05 / dt
        print('pattern_comp', pattern_comp, 'skip', skip)
    else:
        pattern_idx = np.array([], dtype=int)

    balance_index = np.zeros(int(len(time) // skip) + 1)

    # Simulate spikes, filtered activity, and plasticity.
    for t in range(len(time) - 1):
        np.fill_diagonal(W, 0)

        if pulse_num == 1:
            if abs(time[t] - pert_time) < 10 * dt:
                g_elig_E += pulse_amp / tau_r * dt
        elif pulse_num > 1:
            if burst_mask[t]:
                g_elig_E += pulse_amp / tau_r

        if pattern_comp and abs(time[t] - pat_time) < 10 * dt:
            g_elig_E[pattern_idx] += (pat_amp / tau_r) * dt

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

        ext_act = ex_weights @ g_elig_X
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act + ext_act
        idx = np.where(Lam < 0)
        Lam[idx] = 0

        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)
        Espike = np.argwhere(random_numE < Lam[:N_E] * dt)
        Ispike = np.argwhere(random_numI < Lam[N_E:] * dt)

        if t % skip == 0:
            track_gex.append(ext_act)
            track_lam_E[:track_num_n, t // skip] = Lam[:track_num_n]
            track_mean_rates[0, t // skip] = np.mean(Lam[:N_E])
            balance_index[t // skip] = np.abs(np.mean(Lam)) / (np.mean(W[:N_E, :N_E] @ g_elig_E) + np.mean(ext_act))
            track_mean_rates[1, t // skip] = np.mean(Lam[N_E:])

        start_plastic = 5.
        if pulse_num > 1:
            if plastic and time[t] > start_plastic:
                W[Espike[Espike <= N_E], :N_E] += np.tile(eta_ee * B12 * x[0, :N_E].transpose(), (np.count_nonzero(Espike <= N_E), 1))
                W[:N_E, Espike[Espike <= N_E]] += np.tile(eta_ee * B21 * (x[0, :N_E] - alpha_ee), (np.count_nonzero(Espike <= N_E), 1)).transpose()
                W[:N_E, Ispike[Ispike <= N_I] + N_E] -= np.tile(eta_ei * B21_inh * (x[0, :N_E] - alpha_ei), (np.count_nonzero(Ispike <= N_I), 1)).transpose()
                W[Espike[Espike <= N_E], N_E:] -= np.tile(eta_ei * B12_inh * x[0, N_E:].transpose(), (np.count_nonzero(Espike < N_E), 1))

        np.fill_diagonal(W, 0)
        W[:N_E, :N_E][W[:N_E, :N_E] < 0] = 0
        W[:N_E, N_E:][W[:N_E, N_E:] > 0] = 0

        tau_r_I = 2 * tau_r
        x[0, :N_E] -= dt * x[0, :N_E] / tau_STDP
        x[0, N_E:] -= dt * x[0, N_E:] / tau_STDP
        g_elig_E -= dt * g_elig_E / tau_r
        g_elig_I -= dt * g_elig_I / tau_r_I
        g_elig_X -= dt * g_elig_X / tau_r

        idx = np.where(random_numE < Lam[:N_E] * dt)
        x[0, idx] += 1
        if normalize:
            g_elig_E[idx] += 1 / tau_r
        else:
            g_elig_E[idx] += 1

        idx = np.where(random_numI < Lam[N_E:] * dt)
        if normalize:
            g_elig_I[idx] += 1 / tau_r_I
        else:
            g_elig_I[idx] += 1
        idx = N_E + np.array(idx)
        x[0, idx] += 1

        idx = np.where(random_numX < lambda_X * dt)
        if normalize:
            g_elig_X[idx] += 1 / tau_r
        else:
            g_elig_X[idx] += 1

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

            if mean_nonzero_ee > 1 or np.abs(mean_nonzero_ei) > 1:
                track_mean_W[0, t // skip:] = track_mean_W[0, t // skip - 1]
                track_mean_W[1, t // skip:] = track_mean_W[1, t // skip - 1]
                track_W_mean = np.vstack([track_mean_W[0, :], track_mean_W[1, :]])
                time_reduced = time[::skip]
                return time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, balance_index, b

    time_reduced = time[::skip]
    print('reduced time', np.shape(time_reduced))
    track_W_mean = np.vstack([track_mean_W[0, :], track_mean_W[1, :]])
    return time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, balance_index, b

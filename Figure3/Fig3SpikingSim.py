import numpy as np

from helper_functions import calculate_CSD, generate_bernouilli_weight_array, generate_bernouilli_wx, generate_block_wx, generate_weight_array


def spiking_sim(normalize, axs, seed, c_x, sigma, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II, aE=20, dt_theory=None):
    """Run the spiking E-I network and its accompanying population theory."""
    print('c_X', c_x, 'seed', seed, 'taur', tau_r, 'normalize', normalize)
    ms_per_sec = 1000
    Num_neur = N_E + N_I
    time = np.linspace(0, T, int(T / dt))
    if dt_theory is None:
        dt_theory = dt
    theory_substeps = int(round(dt / dt_theory))

    # Plasticity-rule constants.
    B12 = 1 / 2
    B21 = 1 / 2
    b = 20.0
    B12_inh = 1 / 2
    B21_inh = 1 / 2
    alpha_ee = b * 2 * tau_STDP
    alpha_ei = b * 2 * tau_STDP

    # Seed and simulation switches.
    np.random.seed(988 * seed)
    plastic = True
    plot_theory = True
    eta_ee = 1 / tau_wee
    eta_ei = 1 / tau_wei
    sigma_s = sigma
    sigma_p = sigma
    num_per_pool = 1
    N_x = Num_neur * num_per_pool
    print('Nx', N_x)
    base_ex = 20

    # Initialize eligibility traces and recurrent weights.
    x = np.zeros(shape=(1, Num_neur))
    g_elig_E = np.zeros(shape=N_E)
    g_elig_I = np.zeros(shape=N_I)
    g_elig_X = np.zeros(shape=N_x)
    W = np.zeros(shape=(Num_neur, Num_neur))
    init_weights = np.array([[w_EE, w_EI], [w_IE, w_II]])

    # Initialize external-input weights.
    Wxm = np.array([0.04, 0.03]) / np.sqrt(N_E)
    Wx_theory = np.array([[Wxm[0], 0], [0, Wxm[1]]])

    bernouilli = False
    if bernouilli:
        W_mat_init = generate_bernouilli_weight_array(init_weights / 0.1, N_E, N_I)
        Wx, p = generate_bernouilli_wx(Wxm / 0.1, Num_neur, N_x, num_per_pool)
        w_idx_zeros = np.where(W_mat_init == 0.0)
        Wx_theory = Wx_theory / 0.1
    else:
        Wx = generate_block_wx(Wxm, Num_neur, N_x)
        W_mat_init = generate_weight_array(init_weights, N_E, N_I)

    W_mat_init[:, N_E:] = W_mat_init[:, N_E:] * -1
    W[:, :] = W_mat_init

    # Enforce the original signs of the I-to-E and I-to-I blocks.
    idx_ie = np.where(W[N_E:, :N_E] < 0)
    idx_ii = np.where(W[N_E:, N_E:] > 0)
    W[N_E + idx_ie[0], idx_ie[1]] = 0
    W[N_E + idx_ii[0], N_E + idx_ii[1]] = 0

    # Allocate tracked simulation quantities.
    store_rates = True
    store_weights = True
    store_mean_weights = True
    store_theory_lamb = True
    track_num_n = 5
    skip = 20
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

    # Initialize the population-level theory with the same weights.
    if plot_theory:
        W_theory_matr = np.array([[w_EE, -w_EI], [w_IE, -w_II]])
        NW_theory = np.array([[(N_E - 1) * w_EE, -N_I * w_EI], [N_E * w_IE, -(N_I - 1) * w_II]])
        print(w_EE)
        if bernouilli:
            NW_theory = NW_theory / 0.1

    # Store the complete spike history over the requested simulation duration.
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

    track_mean_x = np.zeros(shape=(2, int(len(time) // skip) + 1))
    track_lam_E = np.zeros(shape=(track_num_n, int(len(time) // skip) + 1))
    track_lam_I = np.zeros(shape=10)
    track_mean_rates = np.zeros(shape=(2, int(len(time) // skip) + 1))

    tr = np.array([0, 0])
    tw = np.array([w_EE, -w_EI])
    if bernouilli:
        tw = tw / 0.1
    if plot_theory:
        theory_rates = np.zeros((2, int(len(time) // skip) + 1))
        theory_weights = np.zeros((2, int(len(time) // skip) + 1))
        theory_weights[:, 0] = np.array([w_EE, -w_EI])
        dwdt = np.zeros((2, 2))
        cov_term = np.zeros((2, 2))
    balance_index = None

    aI = base_ex
    ex_weights = Wx

    # Initialize external-rate processes and reseed the dynamical noise.
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

    # Retain the original spike-time bookkeeping.
    last_spike_time_e = np.full(N_E, -np.inf)
    last_spike_time_i = np.full(N_I, -np.inf)

    # -------------------------------------------------------------------------
    # Spiking simulation and population-theory integration
    # -------------------------------------------------------------------------
    for t in range(len(time) - 1):
        np.fill_diagonal(W, 0)
        if bernouilli:
            W[w_idx_zeros] = 0

        # Update the external E and I rate processes.
        if time[t] > 0.0:
            lambda_xe = aE + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pe
            lambda_xi = aI + np.sqrt(c_x) * x_ou_s + np.sqrt(1 - c_x) * x_ou_pi
        else:
            lambda_xe = 0
            lambda_xi = 0

        lambda_X[:N_x // 2] = lambda_xe
        lambda_X[N_x // 2:] = lambda_xi
        if t % skip == 0:
            track_lamx.append(lambda_X[0])

        dx_s = -x_ou_s * dt / tau_ou + (sigma_s / tau_ou) * np.random.normal(0, 1) * np.sqrt(dt)
        x_ou_s += dx_s
        dx_pe = -x_ou_pe * dt / tau_ou + (sigma_p / tau_ou) * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe
        dx_pi = -x_ou_pi * dt / tau_ou + (sigma_p / tau_ou) * np.random.normal(0, 1, size=1).repeat(N_x // 2) * np.sqrt(dt)
        x_ou_pi += dx_pi

        if store_weights and t % skip == 0:
            track_W[:track_num_n, t // skip] = W[x_idx_e, y_idx_e]
            track_W[track_num_n:, t // skip] = W[x_idx_i, N_E + y_idx_i]

        # Compute recurrent and external input intensities.
        ext_act = ex_weights @ g_elig_X
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act + ext_act
        idx = np.where(Lam < 0)
        Lam[idx] = 0

        # Draw E, I, and external spikes.
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

        for e_idx in Espike:
            if e_idx < N_E:
                last_spike_time_e[e_idx] = time[t]
        for i_idx in Ispike:
            if i_idx < N_I:
                last_spike_time_i[i_idx] = time[t]

        # Store downsampled rates, intensities, and selected weights.
        if t % skip == 0:
            track_gex.append(ext_act)
            track_mean_x[0, t // skip] = np.mean(x[0, :N_E])
            track_mean_x[1, t // skip] = np.mean(x[0, N_E:])
            track_lam_E[:track_num_n, t // skip] = Lam[:track_num_n]
            track_mean_rates[0, t // skip] = np.mean(Lam[:N_E])
            track_mean_rates[1, t // skip] = np.mean(Lam[N_E:])

        # Turn plasticity on after the original five-second transient.
        start_plastic = 5.0
        if plastic and time[t] > start_plastic:
            W[Espike[Espike <= N_E], :N_E] += np.tile(eta_ee * B12 * x[0, :N_E].transpose(), (np.count_nonzero(Espike <= N_E), 1))
            W[:N_E, Espike[Espike <= N_E]] += np.tile(eta_ee * B21 * (x[0, :N_E] - alpha_ee), (np.count_nonzero(Espike <= N_E), 1)).transpose()
            W[:N_E, Ispike[Ispike <= N_I] + N_E] -= np.tile(eta_ei * B21_inh * (x[0, :N_E] - alpha_ei), (np.count_nonzero(Ispike <= N_I), 1)).transpose()
            W[Espike[Espike <= N_E], N_E:] -= np.tile(eta_ei * B12_inh * x[0, N_E:].transpose(), (np.count_nonzero(Espike < N_E), 1))

        np.fill_diagonal(W, 0)
        W[:N_E, :N_E][W[:N_E, :N_E] < 0] = 0
        W[:N_E, N_E:][W[:N_E, N_E:] > 0] = 0

        # Decay and increment the eligibility traces.
        tau_r_I = 2 * tau_r
        x[0, :N_E] -= dt * x[0, :N_E] / tau_STDP
        x[0, N_E:] -= dt * x[0, N_E:] / tau_STDP
        g_elig_E -= dt * g_elig_E / tau_r
        g_elig_I -= dt * g_elig_I / tau_r_I
        g_elig_X -= dt * g_elig_X / tau_r

        idx = np.where(random_numE < (Lam[:N_E] * dt))
        x[0, idx] += 1
        if normalize:
            g_elig_E[idx] += 1 / tau_r
        else:
            g_elig_E[idx] += 1

        idx = np.where(random_numI < (Lam[N_E:] * dt))
        if normalize:
            g_elig_I[idx] += 1 / tau_r_I
        else:
            g_elig_I[idx] += 1
        idx = N_E + np.array(idx)
        x[0, idx] += 1

        idx = np.where(random_numX < (lambda_X * dt))
        if normalize:
            g_elig_X[idx] += 1 / tau_r
        else:
            g_elig_X[idx] += 1

        if t % 109000 == 0:
            print('======================')
            print('======================')
            print('t', t, 'time[t]', time[t], 'idx1', idx1)
            if store_theory_lamb and plot_theory:
                print('b', b, 'THEORY LAM', tr[0], 'poisson lamb', np.mean(Lam[:N_E]))
            else:
                print('ACTUAL LAMB', 'poisson labmb', np.mean(Lam[:N_E]), np.mean(Lam[N_E:]), 'track_rates[0,t]')

        # Integrate the population theory using the same update equations.
        if store_theory_lamb and plot_theory:
            rx = np.array([aE, aI])
            tau_r_vec = np.array([tau_r, tau_r_I])
            NW_theory[0, 0] = (N_E - 1) * tw[0]
            NW_theory[0, 1] = N_I * tw[1]

            if plastic and time[t] > start_plastic:
                if time[t] > start_plastic + 2:
                    skip_cov = 5000
                else:
                    skip_cov = 1000

                if t % skip_cov == 0:
                    if bernouilli:
                        cov_term = calculate_CSD(N_E, 2, 2, p * NW_theory, p * Wx_theory, tau_STDP, tau_r, tau_ou, c_x, aE, aI, sigma_p, sigma_s, tr[0], tr[1])
                    else:
                        cov_term = calculate_CSD(N_E, 2, 2, NW_theory, Wx_theory, tau_STDP, tau_r, tau_ou, c_x, aE, aI, sigma_p, sigma_s, tr[0], tr[1])

            for _ in range(theory_substeps):
                track_theory_E_lamb = tr[0]
                track_theory_I_lamb = tr[1]
                r_vec = np.array([tr[0], tr[1]])

                W_theory_matr[0, :] = tw
                NW_theory[0, 0] = (N_E - 1) * tw[0]
                NW_theory[0, 1] = N_I * tw[1]

                if normalize:
                    if bernouilli:
                        drdt = (-r_vec + p * (NW_theory @ r_vec + Num_neur // 2 * Wx_theory @ rx)) / tau_r_vec
                    else:
                        drdt = (-r_vec + NW_theory @ r_vec + Num_neur // 2 * Wx_theory @ rx) / tau_r_vec
                else:
                    drdt = -r_vec / tau_r_vec + NW_theory @ r_vec + Num_neur // 2 * Wx_theory @ rx

                if plastic and time[t] > start_plastic:
                    dwdt[0, 0] = (tau_STDP * track_theory_E_lamb * (track_theory_E_lamb - b) + cov_term[0, 0]) / tau_wee
                    dwdt[0, 1] = -(tau_STDP * track_theory_I_lamb * (track_theory_E_lamb - b) + cov_term[0, 1]) / tau_wei

                tw = tw + dwdt[0, :] * dt_theory
                tr = tr + drdt * dt_theory

            if t % 10000 == 0:
                print('theory_weights[1,t]', tw[1], 'theory_weights[0,t]', tw[0])
                print('##############################################################')
                print('##############################################################')
                print('##############################################################')
                print('theory_rates[:, t]', tr[0], tr[1], 'track_theory_E_lamb', track_theory_E_lamb, 'track_theory_I_lamb', track_theory_I_lamb)
            if t % skip == 0:
                theory_weights[:, t // skip] = tw
                theory_rates[:, t // skip] = tr

        # Store population-average recurrent weights.
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

            if t % 8000 == 0:
                print('track_mean_W_ee[t]', mean_nonzero_ee)
                print('track_mean_W_ei[t]', mean_nonzero_ei)

    time_reduced = time[::skip]

    # Package the theory and simulation outputs in the original return order.
    if plot_theory:
        theory_lamb = np.vstack([theory_rates[0, :], theory_rates[1, :]])
    else:
        theory_weights = np.array([[0, 0], [0, 0]])
        theory_lamb = np.array([[0, 0], [0, 0]])

    track_W_mean = np.vstack([track_mean_W[0, :], track_mean_W[1, :]])
    return time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, theory_weights[0, :], theory_weights[1, :], theory_lamb, balance_index, b

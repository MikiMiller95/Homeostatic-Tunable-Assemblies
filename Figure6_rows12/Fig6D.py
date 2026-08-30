"""Mean-field theory simulation used to generate Figure 6 trajectories."""

import numpy as np

from helper_functions import calculate_CSD


def run_sim(N, w_EE, w_IE, w_II, w_EI, tau_ou, tau_STDP, tau_rprim, tau_r, tau_w_e, tau_w_i, w_EX, w_IX, aE, aI, b, c_x, sigma_s, sigma_p, T):
    """Integrate the two-population rate and plasticity dynamics."""
    ms_per_sec = 1000.
    dt = .2 / ms_per_sec
    time = np.linspace(0, T, int(T / dt))

    # Initialize rate, weight, and plasticity state variables.
    dwdt = np.zeros((2, 2))
    Wx_theory = np.zeros((2, 2))
    Wx_theory[0, 0] = w_EX
    Wx_theory[1, 1] = w_IX
    rates = np.zeros((2, len(time)))
    W = np.zeros((2, len(time)))

    drdt = np.array([0, 0])
    N_E = N
    N_I = N
    NW_theory = np.array([[(N_E - 1) * w_EE, -N_I * w_EI], [N_E * w_IE, -(N_I - 1) * w_II]])
    tw = np.array([w_EE, -w_EI])
    tr = np.array([0, 0])
    tau_r_vec = np.array([tau_r, 2 * tau_r])

    print(',W', tw, 'Wx', Wx_theory, 'tau_ou', tau_ou)
    print('tau_rprim', tau_rprim, 'aE,', aE, 'aI', aI)
    print('tau_r', tau_r, 'Wx,', Wx_theory)

    # Integrate the rate dynamics and turn on plasticity after the transient.
    for i in range(len(time) - 1):
        rE = tr[0]
        rI = tr[1]
        r_vec = np.array([rE, rI])
        rx = np.array([aE, aI])

        NW_theory[0, 0] = (N_E - 1) * tw[0]
        NW_theory[0, 1] = N_I * tw[1]
        drdt = (-r_vec + (NW_theory @ r_vec + N * Wx_theory @ rx)) / tau_r_vec

        start_plastic = 4.
        if time[i] > start_plastic:
            if time[i] > start_plastic + 2:
                skip_cov = 5000
            else:
                skip_cov = 1000

            # Recompute the covariance contribution at the original cadence.
            if i % skip_cov == 0:
                cov_term = calculate_CSD(N, 2, 2, NW_theory, Wx_theory, tau_STDP, tau_r, tau_ou, c_x, aE, aI, sigma_p, sigma_s, rE, rI, normalize=True)

            dwdt[0, 0] = (1 / tau_w_e) * (tau_STDP * rE * (rE - b) + cov_term[0, 0])
            dwdt[0, 1] = -(1 / tau_w_i) * (tau_STDP * rI * (rE - b) + cov_term[0, 1])

        # Euler updates for recurrent weights and population rates.
        tw = tw + dwdt[0, :] * dt
        tr = tr + drdt * dt
        rates[:, i + 1] = rates[:, i] + drdt * dt
        W[:, i + 1] = tw
        rates[:, i + 1] = tr

    return rates, time, np.abs(W)

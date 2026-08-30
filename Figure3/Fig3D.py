import numpy as np


def run_sim(N, w_EE, w_IE, w_II, w_EI, tau_ou, tau_STDP, tau_r, tau_w_e, tau_w_i, w_EX, w_IX, aE, aI, b, c_x, sigma_s, sigma_p, T):
    """Run the two-population rate dynamics used for the stable/unstable comparison."""
    ms_per_sec = 1000.0
    dt = 0.1 / ms_per_sec
    time = np.linspace(0, T, int(T / dt))

    # Initialize rates, recurrent weights, and external input weights.
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
    print(',W', tw, 'Wx', Wx_theory, 'tau_ou', tau_ou)
    print('aE,', aE, 'aI', aI)
    print('tau_r', tau_r, 'Wx,', Wx_theory)
    W[:, 0] = tw
    tau_r_vec = np.array([tau_r, tau_r * 2])

    # Integrate rate and plasticity dynamics with the original update order.
    for i in range(len(time) - 1):
        rE = tr[0]
        rI = tr[1]
        r_vec = np.array([rE, rI])
        rx = np.array([aE, aI])

        NW_theory[0, 0] = (N_E - 1) * tw[0]
        NW_theory[0, 1] = N_I * tw[1]
        drdt = (-r_vec + (NW_theory @ r_vec + N * Wx_theory @ rx)) / tau_r_vec

        start_plastic = 5.0
        if time[i] > start_plastic:
            dwdt[0, 0] = (1 / tau_w_e) * (tau_STDP * rE * (rE - b))
            dwdt[0, 1] = -(1 / tau_w_i) * (tau_STDP * rI * (rE - b))

        tw = tw + dwdt[0, :] * dt
        tr = tr + drdt * dt
        rates[:, i + 1] = rates[:, i] + drdt * dt
        W[:, i + 1] = tw
        rates[:, i + 1] = tr

    return rates, time, np.abs(W)

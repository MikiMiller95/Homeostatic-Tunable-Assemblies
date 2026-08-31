"""Run the Figure 6 sigma-sweep spiking simulations and save CSV outputs.

This file preserves the supplied parameter values, loop order, simulation call,
and output filename conventions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Fig6SpikingSim import spiking_sim

seeds = 20
sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

fig, axs = plt.subplots(2, 4, figsize=(12, 8), dpi=100)
c_x = 0.
axs = None

for i in range(seeds):
    for sigma in sigmas:
        # Simulation size and integration step
        ms_per_sec = 1000
        print('inside')
        N_E = 500
        N_I = 500
        T = 110.
        ms_per_sec = 1000
        dt = .2 / ms_per_sec

        # Synaptic, OU, STDP, and plasticity timescales
        tau_ou = 8. / ms_per_sec
        tau_r = 8. / ms_per_sec
        normalize = True
        tau_rprim = 2.0 / ms_per_sec
        tau_STDP = 120. / ms_per_sec
        tau_wei = 75000
        tau_wee = 225000
        print('tau_wee', tau_wee)
        print('tau_wei', tau_wei)

        Num_neur = N_I + N_E

        # Initial recurrent weights
        factor = np.sqrt(N_E)
        w_IE = 10. / factor * tau_rprim
        w_II = 10. / factor * tau_rprim
        w_EE = 18. / factor * tau_rprim
        w_EI = 18. / factor * tau_rprim
        ctt = 0
        loc = 'Fig6'

        # Run one seed/correlation-strength condition
        time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, Wee_theory, Wei_theory, theory_lamb, balance_index, cval, q_val = spiking_sim(normalize, axs, 35 * i, c_x, sigma, ctt, tau_rprim, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II)

        # Save rates, reciprocity, and mean weights
        track_mean_rates = pd.DataFrame(track_mean_rates)
        track_mean_rates.to_csv(f'{loc}data/mean_rates_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        track_qval = pd.DataFrame(q_val)
        track_qval.to_csv(f'{loc}data/q_val_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        mean_ee_weights = track_W_mean[0, :]
        mean_ei_weights = track_W_mean[1, :]
        df_ee_weights = pd.DataFrame(mean_ee_weights)
        df_ee_weights.to_csv(f'{loc}data/mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_ei_weights = pd.DataFrame(mean_ei_weights)
        df_ei_weights.to_csv(f'{loc}data/mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Save balance and theory outputs returned by the simulation
        balance_index = pd.DataFrame(balance_index)
        balance_index.to_csv(f'{loc}data/balance_idx_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_theory_lamb = pd.DataFrame(theory_lamb)
        df_theory_lamb.to_csv(f'{loc}data/lamb_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_Wee_theory = pd.DataFrame(Wee_theory)
        df_Wee_theory.to_csv(f'{loc}data/W_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_Wei_theory = pd.DataFrame(Wei_theory)
        df_Wei_theory.to_csv(f'{loc}data/Wei_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        del balance_index
        del track_W_mean
        del df_ei_weights
        del df_ee_weights
        del df_Wei_theory
        del df_Wee_theory
        del track_W

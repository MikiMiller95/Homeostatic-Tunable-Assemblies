import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Fig2SpikingSim import spiking_sim


# Sweep settings used by the supplied Figure 2 simulation runner.
seeds = 1
print('hellow')
sigma = 0
w_EEs = np.array([26,25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14])
w_EIs = np.array([30])
c_x = 0.


# Run each seed and initial-weight condition, then save the recorded summaries.
for i in range(seeds):
    if i == 0:
        fig, axs = plt.subplots(1, 4, figsize=(16, 6), dpi=100)
    else:
        axs = None

    for wee in w_EEs:
        for wei in w_EIs:
            print('inside')

            # Network size and integration settings.
            N_E = 500
            N_I = 500
            T = 40.
            ms_per_sec = 1000
            dt = .1 / ms_per_sec

            # Synaptic, input, and plasticity timescales.
            tau_ou = 8. / ms_per_sec
            tau_r = 8. / ms_per_sec
            tau_STDP = 120. / ms_per_sec
            tau_wei = 75000.
            tau_wee = 225000.
            print('tau_wee', tau_wee)
            print('tau_wei', tau_wei)

            # Scale population-level weights into per-connection simulation units.
            normalize = True
            tau_rprim = 2. / ms_per_sec
            factor = np.sqrt(N_E)
            w_IE = .02 / factor
            w_II = .02 / factor
            w_EE = wee / factor * tau_rprim
            w_EI = wei / factor * tau_rprim
            print('(1+N_E*w_II)*np.sqrt(tau_wei/tau_wee)', (1 + N_E * w_II) * np.sqrt(tau_wei / tau_wee))
            print('Nwie', N_E * w_IE)

            ctt = 0
            loc = 'NewFig2data'
            time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, balance_index, cval = spiking_sim(normalize, axs, i, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II)

            # Use the same scaled values as the plotting script when constructing filenames.
            w_EE = np.around(w_EE * factor, 3)
            w_EI = np.around(w_EI * factor, 3)

            # Save the population-rate trajectories.
            track_mean_rates = pd.DataFrame(track_mean_rates)
            track_mean_rates.to_csv(f'{loc}/mean_rates_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            # Save the individually tracked weights.
            track_W = pd.DataFrame(track_W)
            track_W.to_csv(f'{loc}/track_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            # Save the population-mean E-to-E and I-to-E weights.
            mean_ee_weights = track_W_mean[0, :]
            mean_ei_weights = track_W_mean[1, :]
            df_ee_weights = pd.DataFrame(mean_ee_weights)
            df_ee_weights.to_csv(f'{loc}/mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            df_ei_weights = pd.DataFrame(mean_ei_weights)
            df_ei_weights.to_csv(f'{loc}/mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            # Preserve the existing balance-index output, which is currently empty.
            balance_index = pd.DataFrame(balance_index)
            balance_index.to_csv(f'{loc}/balance_idx_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            del balance_index
            del track_W_mean
            del df_ei_weights
            del df_ee_weights
            del track_W

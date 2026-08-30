"""Generate the single-pulse spiking data loaded by Fig5.py."""

import numpy as np
import pandas as pd

from Fig5SpikingSim import spiking_sim


# Simulation sweep retained from the supplied runner.
seed_indices = range(39, 50)
w_EEs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
sigma = 0
c_x = 0.0
data_dir = 'data'

for i in seed_indices:
    axs = None

    for wee in w_EEs:
        print('wee', wee)

        # Network size and simulation duration.
        ms_per_sec = 1000
        N_E = 500
        N_I = 500
        T = 10.0
        dt = 0.1 / ms_per_sec

        # Synaptic, rate, and plasticity time constants.
        tau_ou = 8.0 / ms_per_sec
        tau_r = 8.0 / ms_per_sec
        tau_STDP = 120.0 / ms_per_sec
        tau_wei = 75000.0
        tau_wee = 225000.0
        print('tau_wee', tau_wee)
        print('tau_wei', tau_wei)

        # Firing-rate targets and weight scaling.
        b = 20
        aE = aI = 20.0
        normalize = True
        tau_rprim = 2.0 / ms_per_sec
        factor = np.sqrt(N_E)

        # Fixed synapses and the selected point on the line attractor.
        w_EX = 0.04 / factor
        w_IX = 0.03 / factor
        w_IE = 0.02 / factor
        w_II = 0.02 / factor
        w_EE = wee / factor * tau_rprim
        num = (-b + b * N_E * w_EE + aE * N_E * w_EX) * (1 + N_E * w_II)
        den = b * N_E**2 * w_IE + aI * N_E**2 * w_IX
        w_EI = num / den
        print('wei', w_EE)

        # Run the spiking simulation. The internal random seed remains 400 * i.
        ctt = 0
        _, track_W, track_W_mean, track_mean_rates, _, _, _, balance_index, _ = spiking_sim(normalize, axs, 400 * i, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II, pulse_num=1)

        # Match the filename scaling used by the Figure 5 loader.
        w_EE = np.around(w_EE * factor, 3)
        w_II = np.around(w_II * factor, 3)
        w_EI = np.around(w_EI * factor, 3)
        w_IE = np.around(w_IE * factor, 3)

        # Save population rates and example synaptic-weight traces.
        track_mean_rates = pd.DataFrame(track_mean_rates)
        track_mean_rates.to_csv(f'{data_dir}/one_pulse_mean_rates_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        track_W = pd.DataFrame(track_W)
        track_W.to_csv(f'{data_dir}/one_pulse_track_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Save mean recurrent weights and the balance index.
        mean_ee_weights = track_W_mean[0, :]
        mean_ei_weights = track_W_mean[1, :]

        df_ee_weights = pd.DataFrame(mean_ee_weights)
        df_ee_weights.to_csv(f'{data_dir}/one_pulse_mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_ei_weights = pd.DataFrame(mean_ei_weights)
        df_ei_weights.to_csv(f'{data_dir}/one_pulse_mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        balance_index = pd.DataFrame(balance_index)
        balance_index.to_csv(f'{data_dir}/one_pulse_balance_idx_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Release the largest arrays before moving to the next condition.
        del balance_index
        del track_W_mean
        del df_ei_weights
        del df_ee_weights
        del track_W

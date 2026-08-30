import numpy as np
import pandas as pd

from Fig3D import run_sim
from Fig3SpikingSim import spiking_sim


# Generate the stable and unstable initial-condition rate traces.
seeds = 20
print('hellow')
sigma = 0.0
w_EE_coords = [18, 33]
c_x = 0.0

for i in range(seeds):
    for wee in w_EE_coords:
        # Network and simulation parameters.
        ms_per_sec = 1000
        print('inside')
        N_E = 500
        N_I = 500
        T = 5.0
        dt = 0.1 / ms_per_sec
        tau_ou = 8.0 / ms_per_sec
        tau_r = 8.0 / ms_per_sec
        tau_STDP = 120.0 / ms_per_sec
        tau_wee = 225000.0
        tau_wei = 75000.0
        print('tau_wee', tau_wee)
        print('tau_wei', tau_wei)

        # Scale recurrent and external weights exactly as in the original script.
        normalize = True
        tau_rprim = 2.0 / ms_per_sec
        factor = np.sqrt(N_E)
        w_IE = 10.0 / factor * tau_rprim
        w_II = 10.0 / factor * tau_rprim
        w_EE = wee / factor * tau_rprim
        w_EI = 20.0 / factor * tau_rprim
        w_EX = 20.0 / factor * tau_rprim
        w_IX = 15.0 / factor * tau_rprim
        aE, aI, b = 20, 20, 20.0
        loc = 'RateDynamics'

        # Run and save the spiking rates used by Fig3.py.
        time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, Wee_theory, Wei_theory, theory_lamb, balance_index, cval = spiking_sim(normalize, None, 400 * i, c_x, sigma, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II)

        track_mean_rates = pd.DataFrame(track_mean_rates)
        track_mean_rates.to_csv(f'{loc}data/mean_rates_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        mean_ee_weights = track_W_mean[0, :]
        mean_ei_weights = track_W_mean[1, :]
        df_ee_weights = pd.DataFrame(mean_ee_weights)
        df_ee_weights.to_csv(f'{loc}data/mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_ei_weights = pd.DataFrame(mean_ei_weights)
        df_ei_weights.to_csv(f'{loc}data/mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        balance_index = pd.DataFrame(balance_index)
        balance_index.to_csv(f'{loc}data/balance_idx_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Release the largest spiking arrays before running the rate model.
        del balance_index
        del track_W_mean
        del df_ei_weights
        del df_ee_weights
        del track_W

        # Run and save the deterministic rate-model comparison.
        rates_theory, time_theory, rec_weights_theory = run_sim(N_E, w_EE, w_IE, w_II, w_EI, tau_ou, tau_STDP, tau_r, tau_wee, tau_wei, w_EX, w_IX, aE, aI, b, c_x, sigma, sigma, T)
        print(np.shape(rates_theory))

        re_theory = pd.DataFrame(rates_theory[0, :])
        re_theory.to_csv(f'{loc}data/re_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        ri_theory = pd.DataFrame(rates_theory[1, :])
        ri_theory.to_csv(f'{loc}data/ri_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

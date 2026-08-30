import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Fig3SpikingSim import spiking_sim


# Generate the simulations used to compare attractors across external E input rates.
seeds = 1
print('hellow')
sigma = 0
aEs = [10, 20, 30]
c_x = 0.0

for i in range(seeds):
    if i == 0:
        fig, axs = plt.subplots(1, 4, figsize=(16, 6), dpi=100)
    else:
        axs = None

    for aE in aEs:
        # Select the initial conditions associated with each external E input rate.
        if aE == aEs[0]:
            wEEs = [20, 22, 24]
            wei = 20
        if aE == aEs[1]:
            wEEs = [16, 18, 20, 22, 24, 26]
            wei = 33
        if aE == 30:
            wei = 55
            wEEs = [16, 18, 20, 22, 24, 26]

        for w_EE in wEEs:
            # Network and simulation parameters.
            ms_per_sec = 1000
            print('inside')
            N_E = 500
            N_I = 500
            T = 30.0
            dt = 0.1 / ms_per_sec
            tau_ou = 8.0 / ms_per_sec
            tau_r = 8.0 / ms_per_sec
            tau_STDP = 120.0 / ms_per_sec
            tau_wei = 75000.0
            tau_wee = 225000.0
            print('tau_wee', tau_wee)
            print('tau_wei', tau_wei)

            # Scale the recurrent weights exactly as in the original script.
            normalize = True
            tau_rprim = 2.0 / ms_per_sec
            factor = np.sqrt(N_E)
            w_IE = 10.0 / factor * tau_rprim
            w_II = 10.0 / factor * tau_rprim
            w_EE = w_EE / factor * tau_rprim
            w_EI = wei / factor * tau_rprim
            loc = 'VaryaE'

            # Run one initial condition and retain all original outputs.
            time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, Wee_theory, Wei_theory, theory_lamb, balance_index, cval = spiking_sim(normalize, axs, 400 * i, c_x, sigma, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II, aE=aE)

            # Save the simulation and theory arrays consumed by Fig3.py.
            track_mean_rates = pd.DataFrame(track_mean_rates)
            track_mean_rates.to_csv(f'{loc}/aE{aE}_mean_rates_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            mean_ee_weights = track_W_mean[0, :]
            mean_ei_weights = track_W_mean[1, :]
            df_ee_weights = pd.DataFrame(mean_ee_weights)
            df_ee_weights.to_csv(f'{loc}/aE{aE}_mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            df_ei_weights = pd.DataFrame(mean_ei_weights)
            df_ei_weights.to_csv(f'{loc}/aE{aE}_mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            balance_index = pd.DataFrame(balance_index)
            balance_index.to_csv(f'{loc}/aE{aE}_balance_idx_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            df_Wee_theory = pd.DataFrame(Wee_theory)
            df_Wee_theory.to_csv(f'{loc}/aE{aE}_W_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            df_Wei_theory = pd.DataFrame(Wei_theory)
            df_Wei_theory.to_csv(f'{loc}/aE{aE}_Wei_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            df_theory_lamb = pd.DataFrame(theory_lamb)
            df_theory_lamb.to_csv(f'{loc}/aE{aE}_lamb_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

            # Release the largest arrays before the next initial condition.
            del balance_index
            del track_W_mean
            del df_ei_weights
            del df_ee_weights
            del df_Wei_theory
            del df_Wee_theory
            del track_W

        if i == 0:
            fig.savefig(f'{loc}plots_wee{w_EE}_wei{w_EI}_seed{i}_sigma{sigma}.pdf', dpi=300)
            plt.cla()

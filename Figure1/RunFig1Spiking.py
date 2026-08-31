# Numerical, plotting, simulation, data-export, and profiling dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Fig1SpikingSim import spiking_sim
import pandas as pd
import cProfile
import pstats
import pandas as pd


# =============================================================================
# Simulation sweep
# =============================================================================

seeds = 1 # make bigger if you want to get more seeds ; this is made low for ease of initial tests
print('hellow')
sigmas = [0.]
c_x = 0.
axs = None

# Run seeds 1 through 14 for each requested external-noise amplitude.
for i in range(seeds):
    for sigma in sigmas:
        # Network size, duration, and numerical time step.
        ms_per_sec = 1000
        print('inside')
        N_E = 500
        N_I = 500
        T = 100.
        ms_per_sec = 1000
        dt = .1 / ms_per_sec

        # Synaptic, OU, STDP, and plasticity timescales.
        tau_ou = 8. / ms_per_sec
        tau_r = 8. / ms_per_sec
        tau_STDP = 120. / ms_per_sec
        normalize = True

        factor = np.sqrt(N_E)
        tau_wei = 75000
        tau_wee = 225000
        print('tau_wee', tau_wee)
        print('tau_wei', tau_wei)

        # Initial recurrent weights and simulation bookkeeping.
        Num_neur = N_I + N_E
        w_IE = .02 / factor
        w_II = .02 / factor
        w_EE = .036 / factor
        w_EI = .036 / factor
        ctt = 0
        loc = 'Fig1'

        # Run the spiking simulation without changing its arguments or seed mapping.
        time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, balance_index, cval = spiking_sim(normalize, axs, 4000 * i, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II)

        # Save the population-mean firing rates.
        track_mean_rates = pd.DataFrame(track_mean_rates)
        track_mean_rates.to_csv(f'{loc}data/mean_rates_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Save the mean E-to-E and I-to-E weight trajectories.
        mean_ee_weights = track_W_mean[0, :]
        mean_ei_weights = track_W_mean[1, :]
        df_ee_weights = pd.DataFrame(mean_ee_weights)
        df_ee_weights.to_csv(f'{loc}data/mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_ei_weights = pd.DataFrame(mean_ei_weights)
        df_ei_weights.to_csv(f'{loc}data/mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Save the balance index using the original filename convention.
        balance_index = pd.DataFrame(balance_index)
        balance_index.to_csv(f'{loc}data/balance_idx_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Release the largest per-seed arrays before starting the next run.
        del balance_index
        del track_W_mean
        del df_ei_weights
        del df_ee_weights
        del track_W

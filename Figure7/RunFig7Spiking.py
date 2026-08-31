"""Run the long spiking simulations used for the Figure 7 weight trajectories.

Only comments, spacing, duplicate imports, and clearly unused setup lines were
cleaned. Variable names, capitalization, parameter values, loop ranges,
simulation calls, saved arrays, and CSV filename conventions are unchanged.
"""

import numpy as np
import pandas as pd

from Fig7SpikingSim import spiking_sim


# Correlation sweep and seed range.
seeds=1
print('hellow')
sigma=.5
c_xs = np.array([0.,.2,.4,.6,.8,1.])

for i in range(seeds):
    # The original runner does not pass axes to the simulation.
    axs = None

    for c_x in c_xs:
        print('inside')

        # Network and integration parameters.
        N_E = 500
        N_I = 500
        T =150.
        ms_per_sec = 1000
        dt = .2/ms_per_sec

        # Spike-response and plasticity parameters.
        tau_ou = 8./ms_per_sec
        tau_r =8./ms_per_sec
        normalize = True
        if normalize:
            tau_rprim=2./ms_per_sec
        else:
            tau_rprim=1.

        tau_STDP =120./ms_per_sec
        tau_wei = 75000.
        tau_wee =225000.
        print('tau_wee',tau_wee)
        print('tau_wei',tau_wei)

        # Initial recurrent weights.
        factor =np.sqrt(N_E)
        w_IE=.02/ factor
        w_II=.02 / factor
        w_EE=.036 / factor
        w_EI=.036 / factor
        ctt=0
        loc='Fig7'

        # Run the simulation. The pre-existing argument and return-value
        # inconsistencies with Fig7SpikingSim.py are documented in the README
        # and are intentionally not corrected here.
        time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, spike_trains_E, spike_trains_I, Wee_theory, Wei_theory, theory_lamb, balance_index, cval = spiking_sim(normalize, axs, 35 * i, c_x, sigma, ctt, tau_rprim, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II)
        # Save population rates and the auxiliary q-value output expected by
        # the original runner.
        track_mean_rates = pd.DataFrame(track_mean_rates)
        track_mean_rates.to_csv(f'{loc}data/mean_rates_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Save the mean E-to-E and I-to-E weight trajectories used by the
        # Figure 7 analysis.
        mean_ee_weights = track_W_mean[0,:]
        mean_ei_weights = track_W_mean[1,:]

        df_ee_weights = pd.DataFrame(mean_ee_weights)
        df_ee_weights.to_csv(f'{loc}data/mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_ei_weights = pd.DataFrame(mean_ei_weights)
        df_ei_weights.to_csv(f'{loc}data/mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Save diagnostic and theory trajectories returned by the simulation.
        balance_index = pd.DataFrame(balance_index)
        balance_index.to_csv(f'{loc}data/balance_idx_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_theory_lamb = pd.DataFrame(theory_lamb)
        df_theory_lamb.to_csv(f'{loc}data/lamb_theory_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_Wee_theory = pd.DataFrame(Wee_theory)
        df_Wee_theory.to_csv(f'{loc}data/W_theory_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_Wei_theory = pd.DataFrame(Wei_theory)
        df_Wei_theory.to_csv(f'{loc}data/Wei_theory_Ne_{N_E}_T{T}_wee{w_EE*factor}_wei{w_EI*factor}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        # Release the largest arrays before the next simulation.
        del balance_index
        del track_W_mean
        del df_ei_weights
        del df_ee_weights
        del df_Wei_theory
        del df_Wee_theory
        del track_W

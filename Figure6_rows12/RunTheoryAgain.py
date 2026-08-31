"""Generate the Figure 6 mean-field theory weight trajectories.

The supplied theory parameters, loop order, timescale conversion, and output
filename conventions are unchanged. The theory implementation is provided by
``Fig6D.py``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Fig6D import run_sim

seeds = 20
print('hellow')

sigmas = [.05, .15, .25, .35, .45, .55, .6]
sigmas = [0., .1, .2, .3, .4, .5]

fig, axs = plt.subplots(2, 4, figsize=(12, 8), dpi=100)
c_x = 0.

for i in range(seeds):
    if i == 10 or i == 0:
        fig, axs = plt.subplots(2, 4, figsize=(12, 8), dpi=100)
    else:
        axs = None

    for sigma in sigmas:
        # Theory integration and model parameters
        ms_per_sec = 1000
        print('inside')
        N_E = 500
        N_I = 500
        T = 110.
        ms_per_sec = 1000
        dt = .2 / ms_per_sec
        normalize = True
        if normalize:
            tau_rprim = 2. / ms_per_sec
        else:
            tau_rprim = 1.
        tau_ou = 8. / ms_per_sec
        tau_r = 8. / ms_per_sec
        tau_STDP = 120. / ms_per_sec
        tau_wei = 150. / tau_rprim
        tau_wee = 450. / tau_rprim
        print('tau_wee', tau_wee)
        print('tau_wei', tau_wei)

        Num_neur = N_I + N_E

        # Initial recurrent and external weights
        factor = np.sqrt(N_E)
        w_IE = 10. / factor * tau_rprim
        w_II = 10. / factor * tau_rprim
        w_EE = 18. / factor * tau_rprim
        w_EI = 18. / factor * tau_rprim
        w_EX = 20. / factor * tau_rprim
        w_IX = 20. / factor * tau_rprim
        aE, aI, b = 20, 20, 20.
        c_x = 0.
        ctt = 0
        loc = 'Fig6'

        # Run the theory and restore the filename timescale convention
        rates_theory, time_theory, rec_weights_theory = run_sim(N_E, w_EE, w_IE, w_II, w_EI, tau_ou, tau_STDP, tau_rprim, tau_r, tau_wee, tau_wei, w_EX, w_IX, aE, aI, b, c_x, sigma, sigma, T)
        tau_wei = tau_wei * .002
        tau_wee = tau_wee * .002

        print(np.shape(rec_weights_theory))
        df_Wee_theory = pd.DataFrame(rec_weights_theory[0, :])
        df_Wee_theory.to_csv(f'{loc}data/W_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_Wei_theory = pd.DataFrame(rec_weights_theory[1, :])
        df_Wei_theory.to_csv(f'{loc}data/Wei_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

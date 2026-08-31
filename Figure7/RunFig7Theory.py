"""Generate the deterministic theory trajectories used by Figure 7."""

import numpy as np
import pandas as pd
from Fig7D import run_sim


# Correlation sweep and output seed labels.
seeds=20
print('hellow')
c_xs=[0., .2, .4 , .6, .8, 1.]
sigma=.5

for i in range(seeds):
    for c_x in c_xs:
        print('inside')

        # Network and integration parameters.
        N_E = 500
        N_I = 500
        T =110.
        ms_per_sec = 1000
        dt = .1/ms_per_sec

        # Spike-response and plasticity parameters. The plasticity time
        # constants are divided by tau_rprim for the normalized equations.
        normalize = True
        if normalize:
            tau_rprim=2./ms_per_sec
        else:
            tau_rprim=1.
        tau_ou = 8./ms_per_sec
        tau_r =8./ms_per_sec
        tau_STDP =120./ms_per_sec
        tau_wei = 150. /tau_rprim
        tau_wee =450./tau_rprim
        print('tau_wee',tau_wee)
        print('tau_wei',tau_wei)

        # Initial recurrent and external weights.
        factor =np.sqrt(N_E)
        w_IE=10./ factor * tau_rprim
        w_II=10. / factor * tau_rprim
        w_EE=18. / factor * tau_rprim
        w_EI=18. / factor * tau_rprim
        w_EX = 20. / factor * tau_rprim
        w_IX = 20. / factor * tau_rprim
        aE, aI, b = 20, 20, 20.
        loc='Fig7'

        rates_theory, time_theory, rec_weights_theory = run_sim(N_E, w_EE, w_IE, w_II, w_EI, tau_ou, tau_STDP, tau_r, tau_wee, tau_wei, w_EX, w_IX, aE, aI, b, c_x, sigma, sigma, T,dt)

        # Convert the normalized plasticity constants back to the 450 and 150
        # values used in the Figure 7 filenames.
        tau_wei=tau_wei*tau_rprim
        tau_wee=tau_wee*tau_rprim
        print('shape recs thoery',np.shape(rec_weights_theory))

        df_Wee_theory = pd.DataFrame(rec_weights_theory[0,:])
        df_Wee_theory.to_csv(f'{loc}data/W_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

        df_Wei_theory = pd.DataFrame(rec_weights_theory[1,:])
        df_Wei_theory.to_csv(f'{loc}data/Wei_theory_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

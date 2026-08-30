"""Run the multi-pulse simulation used in the third row of Figure 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from Figure2Row3SpikingSim import spiking_sim


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'Figure2Row3Data'


def main():
    """Run the original multi-pulse condition and save its recorded outputs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Simulation conditions.
    seeds = 1
    sigma = 0
    c_x = 0
    w_EEs = np.array([15])
    w_EIs = np.array([15])

    # Network and timescale parameters.
    ms_per_sec = 1000
    N_E = 500
    N_I = 500
    T = 210.
    dt = .2 / ms_per_sec
    tau_ou = 8. / ms_per_sec
    tau_r = 8. / ms_per_sec
    tau_STDP = 120. / ms_per_sec
    tau_wei = 75000.
    tau_wee = 225000.
    normalize = True
    factor = np.sqrt(N_E)

    for i in range(seeds):
        for wee in w_EEs:
            for wei in w_EIs:
                w_IE = .02 / factor
                w_II = .02 / factor
                w_EE = wee / factor * .002
                w_EI = wei / factor * .002
                Num_neur = N_I + N_E
                ctt = 0

                print('tau_wee', tau_wee)
                print('tau_wei', tau_wei)
                print('(1+N_E*w_II)*np.sqrt(tau_wei/tau_wee)', (1 + N_E * w_II) * np.sqrt(tau_wei / tau_wee))
                print('Nwie', N_E * w_IE)

                # Run the pulse train used for the learning panels.
                time, track_W, track_W_mean, track_mean_rates, time_reduced, track_lam_E, track_lam_I, balance_index, cval = spiking_sim(normalize, None, 40 * i, c_x, sigma, ctt, tau_r, tau_ou, tau_STDP, tau_wee, tau_wei, N_E, N_I, T, dt, w_EE, w_EI, w_IE, w_II, pulse_num=2)

                # Use the original rounded filename tokens.
                w_EE = np.around(w_EE * factor, 3)
                w_II = np.around(w_II * factor, 3)
                w_EI = np.around(w_EI * factor, 3)
                w_IE = np.around(w_IE * factor, 3)

                # Save rates, tracked weights, population-mean weights, and balance.
                track_mean_rates = pd.DataFrame(track_mean_rates)
                track_mean_rates.to_csv(DATA_DIR / f'multi_pulse_mean_rates_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

                track_W = pd.DataFrame(track_W)
                track_W.to_csv(DATA_DIR / f'multi_pulse_track_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

                mean_ee_weights = track_W_mean[0, :]
                mean_ei_weights = track_W_mean[1, :]
                df_ee_weights = pd.DataFrame(mean_ee_weights)
                df_ee_weights.to_csv(DATA_DIR / f'multi_pulse_mean_ee_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

                df_ei_weights = pd.DataFrame(mean_ei_weights)
                df_ei_weights.to_csv(DATA_DIR / f'multi_pulse_mean_ei_weights_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)

                balance_index = pd.DataFrame(balance_index)
                balance_index.to_csv(DATA_DIR / f'balance_idx_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{i}.csv', index=False)


if __name__ == '__main__':
    main()

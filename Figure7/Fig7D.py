"""Two-population rate and plasticity equations used by RunFig7Theory.py."""

import numpy as np

from helper_functions import *


# Integrate the rate and weight dynamics for one value of input correlation.
def run_sim(N, w_EE, w_IE, w_II, w_EI, tau_ou,tau_STDP,tau_r, tau_w_e, tau_w_i, w_EX, w_IX, aE, aI,b, c_x, sigma_s, sigma_p, T,dt):
    time = np.linspace(0, T, int(T/dt))

    # Initialize weight derivatives and external coupling.
    dwdt = np.zeros((2, 2))
    Wx_theory = np.zeros((2,2))
    Wx_theory[0,0] = w_EX
    Wx_theory[1,1] = w_IX

    # Store the rate and recurrent-weight trajectories.
    rates = np.zeros((2, len(time)))
    W = np.zeros((2,len(time)))

    N_E=N
    N_I=N
    NW_theory = np.array([[(N_E-1)*w_EE,-(N_I)*w_EI],[N_E*w_IE,-(N_I-1)*w_II]])
    tw = np.array([w_EE,-w_EI])
    tr = np.array([0,0])

    print(',W',tw,'Wx',Wx_theory,'tau_ou',tau_ou)
    print('aE,',aE,'aI',aI)
    print('tau_r',tau_r,'Wx,',Wx_theory)

    for i in range(len(time) - 1):
        # Update the rate dynamics using the current recurrent weights.
        rE = tr[0]
        rI = tr[1]
        r_vec = np.array([rE, rI])
        rx = np.array([aE, aI])

        NW_theory[0,0] = (N_E-1)*tw[0]
        NW_theory[0,1] = (N_I)*tw[1]
        drdt = (-r_vec + (NW_theory @ r_vec + N*Wx_theory @ rx))/tau_r

        # Turn on plasticity after the initial rate transient. The covariance
        # term is recalculated periodically and reused between updates.
        start_plastic = 4.
        if time[i]>start_plastic:
            if time[i]>start_plastic + 2:
                skip_cov= 5000
            else:
                skip_cov= 1000
            if i%skip_cov==0:
                cov_term = calculate_CSD(N,2,2, NW_theory, Wx_theory,tau_STDP, tau_r,tau_ou,c_x,aE,aI,sigma_p,sigma_s,rE,rI,normalize=True)
            dwdt[0,0] = (1 / tau_w_e) * (tau_STDP*rE*(rE-b)+cov_term[0,0])
            dwdt[0,1] = - (1 / tau_w_i) * (tau_STDP*rI*(rE-b)+cov_term[0,1])

        tw = tw + dwdt[0,:] * dt
        tr = tr + drdt * dt
        W[:,i+1]=tw
        rates[:, i+1] = tr

    return rates,time, np.abs(W)

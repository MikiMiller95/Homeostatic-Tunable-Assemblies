import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from helper_functions import *

# Simulation parameters
def run_sim(N, w_EE, w_IE, w_II, w_EI, tau_ou,tau_STDP,tau_r, tau_w_e, tau_w_i, w_EX, w_IX, aE, aI,b, c_x, sigma_s, sigma_p, T):
	ms_per_sec=1000.

	dt = .1/ms_per_sec  # Time step (sec)
	time = np.linspace(0, T, int(T/dt))

	dwdt = np.zeros((2, 2))
	Wx_theory = np.zeros((2,2))
	Wx_theory[0,0] = w_EX
	Wx_theory[1,1] = w_IX
	#W[:, :, 0] = np.array([[w_EE, -w_EI], [w_IE, -w_II]])
	rates = np.zeros((2, len(time)))
	W = np.zeros((2,len(time)))

	drdt = np.array([0,0])
	N_E=N
	N_I=N
	NW_theory = np.array([[(N_E-1)*w_EE,-(N_I)*w_EI],[N_E*w_IE,-(N_I-1)*w_II]])
	tw = np.array([w_EE,-w_EI])
	tr = np.array([0,0])
	#W[:,0]=tw
	print(',W',tw,'Wx',Wx_theory,'tau_ou',tau_ou)
	print('aE,',aE,'aI',aI)
	print('tau_r',tau_r,'Wx,',Wx_theory)
	W[:,0]=tw
	tau_r_vec=np.array([tau_r,tau_r*2])
	for i in range(len(time) - 1):
		rE = tr[0] #rates[0, i]
		rI = tr[1] #rates[1, i]
		r_vec = np.array([rE, rI])
		rx = np.array([aE, aI]) 
		
		NW_theory[0,0] = (N_E-1)*tw[0]#*Num_neur/2
		NW_theory[0,1] = (N_I)*tw[1]#*Num_neur/2

		drdt = (-r_vec + (NW_theory @ r_vec + N*Wx_theory @ rx))/tau_r_vec

		start_plastic = 5.
		if time[i]>start_plastic:
			dwdt[0,0] = (1 / tau_w_e) * (tau_STDP*rE*(rE-b))
			dwdt[0,1] = - (1 / tau_w_i) * (tau_STDP*rI*(rE-b))

		tw = tw + dwdt[0,:] * dt
		tr = tr + drdt * dt 
		rates[:, i + 1] = rates[:, i] + drdt * dt
		
		W[:,i+1]=tw
		rates[:, i+1] = tr
	return rates,time, np.abs(W)



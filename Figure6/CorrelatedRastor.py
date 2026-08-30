"""Generate spike-train CSV files for the correlated-input raster in Figure 6.

The short simulation, correlation-onset time, spike generation, and output
filenames are unchanged.
"""

import time as TimeTime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import generate_block_wx, generate_weight_array

save = True

# ---------------------------------------------------------------------------
# Short raster simulation
def spiking_sim(seed,c_x,sigma,ctt,tau_r, tau_ou,tau_STDP, tau_wee, tau_wei, N_E,N_I,T, dt, w_EE,w_EI,w_IE,w_II):
    print('c_X',c_x,'seed',seed)
    ms_per_sec = 1000
    Num_neur = N_E+N_I
    ctt=0
    ms_per_sec = 1000
    Num_neur = N_E+N_I
    burnin=0.
    time = np.linspace(0,T,int(T/dt))
    B12 = 1/(2)
    B21 = 1/(2)

    b=20.
    B12_inh = 1/(2)
    B21_inh = 1/(2)

    alpha_ee = b*2*tau_STDP
    alpha_ei =b*2*tau_STDP

    start_time = TimeTime.time()

    np.random.seed(988*seed)
    plastic=True
    # Simulation controls and external-input parameters
    eta_ee = 1/tau_wee
    eta_ei = 1/tau_wei
    sigma_s = sigma
    sigma_p = sigma
    num_per_pool=1
    N_x = Num_neur*num_per_pool
    print('Nx',N_x)
    base_ex = 40

    # State and recording arrays
    x = np.zeros(shape=(1,Num_neur))
    g_elig_E = np.zeros(shape=(N_E))
    g_elig_I = np.zeros(shape=(N_I))
    g_elig_X = np.zeros(shape=(N_x))
    window_size = int((300/ms_per_sec)/dt)


    # -----------------------------------------------------------------------
    # Build recurrent and feedforward connectivity
    W=np.zeros(shape=(Num_neur,Num_neur))
    init_weights=np.array([[w_EE,w_EI],[w_IE,w_II]])

    Wxm=tau_rprim*np.array([30.,30. ])/np.sqrt(N_E)

    Wx = generate_block_wx(Wxm, Num_neur,N_x)

    W_mat_init  = generate_weight_array(init_weights, N_E,N_I)
    W_mat_init[:,N_E:]= W_mat_init[:,N_E:]*-1
    W[:,:] = W_mat_init

    idx_ie = np.where(W[N_E:,:N_E] < 0)
    idx_ii = np.where(W[N_E:,N_E:] > 0)
    W[N_E+idx_ie[0],idx_ie[1]]=0
    W[N_E+idx_ii[0],N_E+idx_ii[1]]=0

    store_rates=True
    store_weights=True
    store_mean_weights=True
    track_num_n = N_E
    skip = 1
    track_mean_W=np.zeros(shape=(2,int(len(time)//skip)+1))
    if store_weights:
        track_W=np.zeros(shape=(2*track_num_n,int(len(time)//skip)+1))

        neur_idx = np.arange(N_E*N_E)
        random_weights_to_store = np.random.choice(neur_idx,track_num_n)
        x_idx_e = random_weights_to_store//N_E
        y_idx_e = random_weights_to_store%N_E

        neur_idx = np.arange(N_E*N_I)
        random_weights_to_store = np.random.choice(neur_idx,track_num_n)
        x_idx_i = random_weights_to_store//N_E
        y_idx_i = random_weights_to_store%N_I
    else:
        track_W=np.array([0,0,0])

    bins_store = int(T/dt)
    if store_rates:
        spike_trains_E =np.zeros((N_E, bins_store))
        spike_trains_I =np.zeros((N_I, bins_store))
        spike_trains_X =np.zeros((N_x, bins_store))
        print('CHECK HERE',N_E,bins_store)
    else:
        spike_trains_E =0
        spike_trains_I =0
        spike_trains_X =0

    track_mean_x = np.zeros(shape=(2,int(len(time)//skip)+1))
    track_rates = np.zeros(shape=(10,len(time)))
    track_mean_rates = np.zeros(shape=(2,int(len(time)//skip)+1))


    tr = np.array([0,0])
    tw = np.array([0,0])
    balance_index = np.zeros(int(len(time)//skip)+1)

    aE = base_ex
    aI = base_ex

    ex_weights = Wx
    ex_weights_E = ex_weights[:N_E,:N_x//2]
    ex_weights_I = ex_weights[N_E:,N_x//2:]


    lambda_X=np.zeros(shape=(N_x))
    lambda_xe=np.zeros(shape=(N_x//2))
    lambda_xi=np.zeros(shape=(N_x//2))
    x_ou_s = 0
    x_ou_pe = np.zeros(shape=(N_x//2))
    x_ou_pi = np.zeros(shape=(N_x//2))
    np.random.seed(94*seed)
    np.fill_diagonal(W,0)
    track_lamx=[]

    # -----------------------------------------------------------------------
    # Time stepping and correlation onset at 6.8 s
    eta_x = []
    for t in range(len(time)-1):
        np.fill_diagonal(W,0)

        if time[t]>.1:
            lambda_xe = (aE + np.sqrt(c_x)*x_ou_s+ np.sqrt(1 - c_x)*x_ou_pe)
            lambda_xi = (aI + np.sqrt(c_x)*x_ou_s + np.sqrt(1 - c_x)*x_ou_pi)
        else:
            lambda_xe=0
            lambda_xi=0

        lambda_X[:N_x//2]=lambda_xe
        lambda_X[N_x//2:]=lambda_xi
        track_lamx.append(lambda_X[0])

        if time[t]<6.800:
            sigma_p=0
            sigma_s=0
        else:
            sigma_p=sigma
            sigma_s=sigma

        eta_x.append(x_ou_pe)

        dx_s = -x_ou_s*dt/tau_ou + (sigma_s/tau_ou)*np.random.normal(0, 1)*np.sqrt(dt)
        x_ou_s+=dx_s

        dx_pe = -x_ou_pe*dt/tau_ou + (sigma_p/tau_ou)*np.random.normal(0, 1,size=1).repeat(N_x//2)*np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe

        dx_pi = -x_ou_pi*dt/tau_ou + (sigma_p/tau_ou)*np.random.normal(0, 1,size=1).repeat(N_x//2)*np.sqrt(dt)
        x_ou_pi +=dx_pi

        if store_weights and t%skip==0:
            track_W[:track_num_n,t//skip]=W[x_idx_e,y_idx_e]
            track_W[track_num_n:,t//skip]=W[x_idx_i,N_E+y_idx_i]

        ext_act = ex_weights @ g_elig_X
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act+ext_act
        if t%skip==0:
            balance_index[t//skip]=np.abs(np.mean(Lam))/(np.mean(W[:N_E,:N_E]@g_elig_E)+np.mean(ext_act))

        idx = np.where(Lam<0)
        Lam[idx]=0
        # Draw E, I, and external spikes from the current intensities
        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)

        if t < bins_store:
            if store_rates:
                spike_trains_E[random_numE < (Lam[:N_E] * dt), t] = time[t]
                spike_trains_I[random_numI < (Lam[N_E:] * dt), t] = time[t]
                spike_trains_X[random_numX < (lambda_X * dt), t] = time[t]
            idx1 = t
        else:
            if store_rates:
                spike_trains_E[:, :-1] = spike_trains_E[:, 1:]
                spike_trains_I[:, :-1] = spike_trains_I[:, 1:]
                spike_trains_X[:, :-1] = spike_trains_X[:, 1:]
                spike_trains_E[random_numE > (Lam[:N_E]* dt), -1] = 0
                spike_trains_I[random_numI > (Lam[N_E:] * dt), -1] = 0
                spike_trains_X[random_numX > (lambda_X * dt), -1] = 0

                spike_trains_E[random_numE < (Lam[:N_E] * dt), -1] = time[t]
                spike_trains_I[random_numI < (Lam[N_E:] * dt), -1] = time[t]
                spike_trains_X[random_numX < (lambda_X * dt), -1] = time[t]
            idx1 = bins_store-1

        Espike=np.argwhere(random_numE < (Lam[:N_E] * dt))
        Ispike=np.argwhere(random_numI < (Lam[N_E:] * dt))

        mean_re=(np.mean(x[0,:N_E])/tau_STDP)
        mean_ri=(np.mean(x[0,N_E:])/tau_STDP)

        if t%skip==0:
            track_mean_x[0,t//skip]=np.mean(x[0,:N_E])
            track_mean_x[1,t//skip]=np.mean(x[0,N_E:])

            track_mean_rates[0,t//skip] = np.mean(Lam[:N_E])
            track_mean_rates[1,t//skip] = np.mean(Lam[N_E:])

        if t%2000==0:
            print(time[t],np.mean(Lam[:N_E]))
        # Remove autapses before updating synaptic traces
        np.fill_diagonal(W,0)

        tau_r_I = 2*tau_r
        x[0,:N_E]-=dt*x[0,:N_E]/tau_STDP
        x[0,N_E:]-=dt*x[0,N_E:]/tau_STDP
        g_elig_E-=dt*g_elig_E/tau_r
        g_elig_I-=dt*g_elig_I/tau_r_I
        g_elig_X-=dt*g_elig_X/tau_r

        # Update excitatory spike and synaptic traces
        idx =np.where(random_numE < (Lam[:N_E] * dt))
        x[0,idx] += 1
        g_elig_E[idx] += 1/tau_r

        # Update inhibitory spike and synaptic traces
        idx =np.where(random_numI < (Lam[N_E:] * dt))
        g_elig_I[idx] += 1/(tau_r_I)
        idx = N_E+np.array(idx)
        x[0,idx] += 1

        # Update external-input synaptic traces
        idx =np.where(random_numX < (lambda_X * dt))
        g_elig_X[idx] += 1/tau_r

    time_reduced = time[::skip]

    plot=False
    if plot:
        fig, ax = plt.subplots(1,2,figsize=(8,8))
        raster_ax = ax[0]
        offset_X = 2 * N_E
        offset_E = 0
        offset_I = N_E
        start=6.75
        end = 6.85
        for i in range(spike_trains_X.shape[0]):
            spikes = spike_trains_X[i,:]
            spikes = spikes[(spikes > start) & (spikes < end)]
            raster_ax.scatter(spikes, np.full_like(spikes, offset_X + i), color="k", marker="|", s=0.15, rasterized=True)

        for i in range(N_E):
            spikes = spike_trains_E[i,:]
            spikes = spikes[(spikes > start) & (spikes < end)]
            raster_ax.scatter(spikes, np.full_like(spikes, offset_E + i), color="tab:red", marker="|", s=0.15, rasterized=True)

        for i in range(N_I):
            spikes = spike_trains_I[i,:]
            spikes = spikes[(spikes > start) & (spikes < end)]
            raster_ax.scatter(spikes, np.full_like(spikes, offset_I + i), color="tab:blue", marker="|", s=0.15, rasterized=True)


        ax[1].plot(time[:-1],np.array(eta_x))

        lbl_ft=8
        title_ft=8
        lt=1

        raster_ax.set_title(rf"Rastir Plot ($\sigma =$ ${sigma}$)", fontsize=title_ft)
        raster_ax.axvline(6.8,color='k',linewidth=lt-.5)
        raster_ax.set_xlim(start, end)
        ax[1].set_xlim(start, end)
        raster_ax.set_xticks([start, end],[start, end],fontsize=lbl_ft)
        ax[1].set_xticks([start, end],[start, end],fontsize=lbl_ft)
        raster_ax.set_xticks([6.8],[6.8],fontsize=lbl_ft)
        ax[1].set_xticks([6.8],[6.8],fontsize=lbl_ft)
        raster_ax.set_xlabel("Time (s)", fontsize=lbl_ft)
        raster_ax.set_ylabel("Neuron Index", fontsize=lbl_ft)
        plt.show()
    return time ,track_rates,track_mean_rates,time_reduced,spike_trains_E, spike_trains_I,spike_trains_X, b

# ---------------------------------------------------------------------------
# Run the raster simulation and save spike trains
print('hellow')
sigmas=[4.]
c_x=0.
for sigma in sigmas:
    ms_per_sec = 1000

    print('inside',sigma)
    # Raster simulation parameters
    N_E = 500
    N_I = 500
    T =7.
    ms_per_sec = 1000
    dt = .2/ms_per_sec
    # Synaptic and plasticity timescales
    tau_ou = 8./ms_per_sec
    tau_r =8./ms_per_sec
    tau_STDP =150./ms_per_sec
    tau_wei = 150.
    tau_wee =450
    print('tau_wee',tau_wee)
    print('tau_wei',tau_wei)

    Num_neur = N_I+N_E

    normalize = True
    if normalize:
        tau_rprim=2./ms_per_sec
    else:
        tau_rprim=1.
    factor =np.sqrt(N_E)
    w_IE=10./ factor * tau_rprim
    w_II=10. / factor * tau_rprim
    w_EE=18. / factor * tau_rprim
    w_EI=18. / factor * tau_rprim
    ctt=0
    loc='Driftdata/'

    c_x = 0.
    seed=54000
    time, track_rates,track_mean_rates, time_reduced,spike_trains_E, spike_trains_I,spike_trains_X, cval= spiking_sim(seed,c_x,sigma,ctt,tau_r, tau_ou,tau_STDP, tau_wee, tau_wei,N_E,N_I,T,dt,w_EE,w_EI,w_IE,w_II)
    # Convert the spike arrays to tabular output
    df_spikes_E = pd.DataFrame(spike_trains_E)
    df_spikes_I = pd.DataFrame(spike_trains_I)
    df_spikes_X = pd.DataFrame(spike_trains_X)

    # Save the external, excitatory, and inhibitory spike trains
    df_spikes_E.to_csv( f'{loc}spike_trains_E_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv', index=False)

    df_spikes_I.to_csv( f'{loc}spike_trains_I_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv', index=False)

    df_spikes_X.to_csv( f'{loc}spike_trains_X_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv', index=False)

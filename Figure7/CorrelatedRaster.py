
"""Generate the correlation-switch raster data used by Figure 7.

Variable names, parameter values, random-number draws, update order, returned
arrays, optional diagnostic plotting block, and CSV filename conventions are
unchanged from the uploaded file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import *



# Short spiking simulation used for the raster panel.
def spiking_sim(seed,c_x,sigma,ctt,tau_r, tau_ou,tau_STDP, tau_wee, tau_wei, N_E,N_I,T, dt, w_EE,w_EI,w_IE,w_II):
    print('c_X',c_x,'seed',seed)
    Num_neur = N_E+N_I
    time = np.linspace(0,T,int(T/dt)) #np.arange(0, T, dt)
    b=20.#cov11/2-cov12/39 #15.
    np.random.seed(988*seed)
    sigma_s = sigma #600.
    sigma_p = sigma  #600.
    num_per_pool=1
    N_x = Num_neur*num_per_pool
    print('Nx',N_x)
    base_ex = 40 #15 #*Num_neur//2 #*np.sqrt(Num_neur) #1000

    # Initialize spike-history and synaptic-response traces.
    x = np.zeros(shape=(1,Num_neur))
    g_elig_E = np.zeros(shape=(N_E))
    g_elig_I = np.zeros(shape=(N_I))
    g_elig_X = np.zeros(shape=(N_x))


    # Initialize recurrent and external weights.
    W=np.zeros(shape=(Num_neur,Num_neur))
    init_weights=np.array([[w_EE,w_EI],[w_IE,w_II]])

    #Initialize JX weights
    Wxm=tau_rprim*np.array([30.,30. ])/np.sqrt(N_E) #np.sqrt(N_E)#/(tau_r)
    
    Wx = generate_block_wx(Wxm, Num_neur,N_x)

    W_mat_init  = generate_weight_array(init_weights, N_E,N_I)
    W_mat_init[:,N_E:]= W_mat_init[:,N_E:]*-1
    W[:,:] = W_mat_init #np.zeros(shape=(Num_neur,Num_neur))#W_mat_init

    idx_ie = np.where(W[N_E:,:N_E] < 0)
    idx_ii = np.where(W[N_E:,N_E:] > 0)
    W[N_E+idx_ie[0],idx_ie[1]]=0
    W[N_E+idx_ii[0],N_E+idx_ii[1]]=0

    store_rates=True
    skip = 1
    bins_store = int(T/dt) #int(2*tau_wee/dt) 
    if store_rates:
        spike_trains_E =np.zeros((N_E, bins_store))
        spike_trains_I =np.zeros((N_I, bins_store))
        spike_trains_X =np.zeros((N_x, bins_store))
        print('CHECK HERE',N_E,bins_store)
    else:
        spike_trains_E =0
        spike_trains_I =0
        spike_trains_X =0

    track_rates = np.zeros(shape=(10,len(time)))
    track_mean_rates = np.zeros(shape=(2,int(len(time)//skip)+1))

    aE = base_ex
    aI = base_ex

    ex_weights = Wx  
    lambda_X=np.zeros(shape=(N_x))
    x_ou_s = 0 #np.zeros(shape=len(time_steps)+1)
    x_ou_pe = np.zeros(shape=(N_x//2))
    x_ou_pi = np.zeros(shape=(N_x//2))
    np.random.seed(904*seed)
    np.fill_diagonal(W,0)
    eta_x = []

    # Advance the OU inputs, network spikes, and response traces.
    for t in range(len(time)-1):#dw=0
        np.fill_diagonal(W,0)

        if time[t]>.1:
            lambda_xe = (aE + np.sqrt(c_x)*x_ou_s+ np.sqrt(1 - c_x)*x_ou_pe)
            lambda_xi = (aI + np.sqrt(c_x)*x_ou_s + np.sqrt(1 - c_x)*x_ou_pi)
        else:
            lambda_xe=0
            lambda_xi=0

        lambda_X[:N_x//2]=lambda_xe
        lambda_X[N_x//2:]=lambda_xi
        if time[t]<6.800:
            c_x=0
        else:
            c_x=1


        eta_x.append(x_ou_pe)

        dx_s = -x_ou_s*dt/tau_ou + (sigma_s/tau_ou)*np.random.normal(0, 1)*np.sqrt(dt)
        x_ou_s+=dx_s

        dx_pe = -x_ou_pe*dt/tau_ou + (sigma_p/tau_ou)*np.random.normal(0, 1,size=1).repeat(N_x//2)*np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe

        dx_pi = -x_ou_pi*dt/tau_ou + (sigma_p/tau_ou)*np.random.normal(0, 1,size=1).repeat(N_x//2)*np.sqrt(dt)
        x_ou_pi +=dx_pi

        ext_act = ex_weights @ g_elig_X #np.matmul(ex_weights,convolved_gX)
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act+ext_act
        idx = np.where(Lam<0)
        Lam[idx]=0
        # Generate spikes
        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)   
        
        if store_rates:
            spike_trains_E[random_numE < (Lam[:N_E] * dt), t] = time[t]
            spike_trains_I[random_numI < (Lam[N_E:] * dt), t] = time[t]
            spike_trains_X[random_numX < (lambda_X * dt), t] = time[t]
        if t%skip==0:
            track_mean_rates[0,t//skip] = np.mean(Lam[:N_E])
            track_mean_rates[1,t//skip] = np.mean(Lam[N_E:])

        if t%2000==0:
            print(time[t],np.mean(Lam[:N_E]))
        #remove autapses
        np.fill_diagonal(W,0)

        tau_r_I = 2*tau_r
        x[0,:N_E]-=dt*x[0,:N_E]/tau_STDP
        x[0,N_E:]-=dt*x[0,N_E:]/tau_STDP
        g_elig_E-=dt*g_elig_E/tau_r
        g_elig_I-=dt*g_elig_I/tau_r_I
        g_elig_X-=dt*g_elig_X/tau_r

        # use the exponential kernel as an eligibility trace like term
        idx =np.where(random_numE < (Lam[:N_E] * dt))
        x[0,idx] += 1  # add spikes to elig trace
        g_elig_E[idx] += 1/tau_r  # add spikes to elig trace

        # use the exponential kernel as an eligibility trace like term
        idx =np.where(random_numI < (Lam[N_E:] * dt))
        g_elig_I[idx] += 1/(tau_r_I)  # add spikes to elig trace
        idx = N_E+np.array(idx)
        x[0,idx] += 1  # add spikes to elig trace

        # use the exponential kernel as an eligibility trace like term
        idx =np.where(random_numX < (lambda_X * dt))
        g_elig_X[idx] += 1/tau_r  # add spikes to elig trace

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

# Parameters used to generate the raster CSV files loaded by Fig7.py.
print('hellow')
sigmas=[4.]
c_x=0.
for sigma in sigmas:
    ms_per_sec = 1000
    print('inside',sigma)
    # Generate the asymmetrical causal STDP window
    N_E = 1000  # Number of excitatory neurons
    N_I = 1000# Number of inhibitory neurons
    T =7.5# Total simulation time in ms
    dt = .1/ms_per_sec     # Time step in ms
    # Spike response parameters
    tau_ou = 8./ms_per_sec
    tau_r =8./ms_per_sec
    tau_STDP =120./ms_per_sec
    tau_wei = 75000.
    tau_wee =225000 
    print('tau_wee',tau_wee)
    print('tau_wei',tau_wei)
    
    normalize = True
    if normalize:
        tau_rprim=2./ms_per_sec
    else:
        tau_rprim=1.
    factor =np.sqrt(N_E) #//2.5 #5*np.sqrt(N_E) # np.sqrt(N_E)
    w_IE=10./ factor * tau_rprim
    w_II=10. / factor * tau_rprim
    w_EE=10. / factor * tau_rprim
    w_EI=10. / factor * tau_rprim
    ctt=0
    loc='Driftdata/'

    c_x = 0.
    seed=530
    time, track_rates,track_mean_rates, time_reduced,spike_trains_E, spike_trains_I,spike_trains_X, cval= spiking_sim(int(seed),c_x,sigma,ctt,tau_r, tau_ou,tau_STDP, tau_wee, tau_wei,N_E,N_I,T,dt,w_EE,w_EI,w_IE,w_II)
    # Convert spike arrays to DataFrames
    df_spikes_E = pd.DataFrame(spike_trains_E)
    df_spikes_I = pd.DataFrame(spike_trains_I)
    df_spikes_X = pd.DataFrame(spike_trains_X)

    # Save to CSV using the same style as your existing filenames
    # Note that for large N_E, these spike files may become huge!
    df_spikes_E.to_csv(f'{loc}spike_trains_E_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv', index=False)
    
    df_spikes_I.to_csv(f'{loc}spike_trains_I_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv', index=False)
    
    df_spikes_X.to_csv(f'{loc}spike_trains_X_Ne_{N_E}_T{T}_wee{w_EE}_wei{w_EI}_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv', index=False)



"""Run the spiking network and plasticity dynamics used for Figure 7.

The function signature, variable names, parameter notation, equations, random
draws, update order, returned arrays, and plotting calls are unchanged. Only
duplicate imports, inactive commented code, and demonstrably unused diagnostic
variables were removed, and explanatory comments were added.
"""

import numpy as np
from helper_functions import *


# Spiking-network simulation for one seed and input-correlation value.
def spiking_sim(normalize, axs,seed,c_x,sigma,ctt,tau_rprim, tau_r, tau_ou,tau_STDP, tau_wee, tau_wei, N_E,N_I,T, dt, w_EE,w_EI,w_IE,w_II):
    print('c_X',c_x,'seed',seed,'taur',tau_r,'normalize',normalize)
    ms_per_sec = 1000
    Num_neur = N_E+N_I
    time = np.linspace(0,T,int(T/dt)) #np.arange(0, T, dt)
    B12 = 1/(2) #1.
    B21 = 1/(2) #1/Aplus - B12 

    b=20.#cov11/2-cov12/39 #15.
    B12_inh = 1/(2)
    B21_inh = 1/(2)

    alpha_ee = b*2*tau_STDP#/B21
    alpha_ei =b*2*tau_STDP


    np.random.seed(988*seed)
    plastic=True
    plot_theory = True
    # Plot spike trains
    eta_ee = 1/tau_wee
    eta_ei = 1/tau_wei
    sigma_s = sigma #600.
    sigma_p = sigma  #600.
    num_per_pool=1
    N_x = Num_neur*num_per_pool
    print('Nx',N_x)
    base_ex = 20 #15 #*Num_neur//2 #*np.sqrt(Num_neur) #1000

    # Initialize the spike-history and synaptic-response traces.
    x = np.zeros(shape=(1,Num_neur))
    g_elig_E = np.zeros(shape=(N_E))
    g_elig_I = np.zeros(shape=(N_I))
    g_elig_X = np.zeros(shape=(N_x))


    # Initialize recurrent and external weights.
    W=np.zeros(shape=(Num_neur,Num_neur))
    init_weights=np.array([[w_EE,w_EI],[w_IE,w_II]])

    Wxm=tau_rprim*np.array([20.,20. ])/np.sqrt(N_E) #np.sqrt(N_E)#/(tau_r)
    Wx_theory=np.array([[Wxm[0],0 ],[0,Wxm[1]]]) #np.sqrt(N_E)#/(tau_r)

    bernouilli=False
    if bernouilli:
        W_mat_init  = generate_bernouilli_weight_array(init_weights/.1, N_E,N_I)
        Wx,p=generate_bernouilli_wx(Wxm/.1, Num_neur,N_x,num_per_pool)

        w_idx_zeros= np.where(W_mat_init==0.)
        Wx_theory=Wx_theory/.1
    else:
        Wx = generate_block_wx(Wxm, Num_neur,N_x)

        W_mat_init  = generate_weight_array(init_weights, N_E,N_I)

    W_mat_init[:,N_E:]= W_mat_init[:,N_E:]*-1
    W[:,:] = W_mat_init #np.zeros(shape=(Num_neur,Num_neur))#W_mat_init

    idx_ie = np.where(W[N_E:,:N_E] < 0)
    idx_ii = np.where(W[N_E:,N_E:] > 0)
    W[N_E+idx_ie[0],idx_ie[1]]=0
    W[N_E+idx_ii[0],N_E+idx_ii[1]]=0

    store_rates=True
    store_weights=True
    store_mean_weights=True
    store_theory_lamb=True
    track_num_n = 5#N_E//20
    skip = 2000
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

    if plot_theory:
        NW_theory = np.array([[(N_E-1)*w_EE,-(N_I)*w_EI],[N_E*w_IE,-(N_I-1)*w_II]])
        print(w_EE)
        if bernouilli:
            NW_theory=NW_theory/.1
        #Only store the most recent bins up to some number of time constants as designated by initializing bins_store

    bins_store =  int(T/dt) #int(2*tau_wee/dt) 
    if store_rates:
        spike_trains_E =np.zeros((N_E, bins_store))
        spike_trains_I =np.zeros((N_I, bins_store))
        spike_trains_X =np.zeros((N_x, bins_store))
        print('CHECK HERE',N_E,bins_store)
    else:
        spike_trains_E =0
        spike_trains_I =0
        spike_trains_X =0

    track_lam_E = np.zeros(shape=(track_num_n,int(len(time)//skip)+1))
    track_lam_I = np.zeros(shape=10) #np.zeros(shape=(track_num_n,int(len(time)//skip)+1))

    track_mean_rates = np.zeros(shape=(2,int(len(time)//skip)+1))


    tr = np.array([0,0])
    tw = np.array([w_EE,-w_EI])
    if bernouilli:
        tw=tw/.1
    if plot_theory:
        theory_rates = np.zeros((2, int(len(time)//skip)+1))
        theory_weights = np.zeros((2, int(len(time)//skip)+1))
        theory_weights[:,0] = np.array([w_EE,-w_EI])
        dwdt = np.zeros((2,2))
        cov_term = np.zeros((2,2))
    balance_index=None

    aE = base_ex
    aI = base_ex

    ex_weights = Wx  


    lambda_X=np.zeros(shape=(N_x))
    lambda_xe=np.zeros(shape=(N_x//2))
    lambda_xi=np.zeros(shape=(N_x//2))
    x_ou_s = 0 #np.zeros(shape=len(time_steps)+1)
    x_ou_pe = np.zeros(shape=(N_x//2))
    x_ou_pi = np.zeros(shape=(N_x//2))
    np.random.seed(9*seed)
    np.fill_diagonal(W,0)






    # Advance the network, plasticity, and matched theory in time.
    for t in range(len(time)-1):#dw=0
        np.fill_diagonal(W,0)
        if bernouilli:
            W[w_idx_zeros]=0

        if time[t]>0.:
            lambda_xe = (aE + np.sqrt(c_x)*x_ou_s+ np.sqrt(1 - c_x)*x_ou_pe)
            lambda_xi = (aI + np.sqrt(c_x)*x_ou_s + np.sqrt(1 - c_x)*x_ou_pi)
        else:
            lambda_xe=0
            lambda_xi=0

        lambda_X[:N_x//2]=lambda_xe
        lambda_X[N_x//2:]=lambda_xi
        dx_s = -x_ou_s*dt/tau_ou + (sigma_s/tau_ou)*np.random.normal(0, 1)*np.sqrt(dt)
        x_ou_s+=dx_s

        dx_pe = -x_ou_pe*dt/tau_ou + (sigma_p/tau_ou)*np.random.normal(0, 1,size=1).repeat(N_x//2)*np.sqrt(dt)
        x_ou_pe = x_ou_pe + dx_pe

        dx_pi = -x_ou_pi*dt/tau_ou + (sigma_p/tau_ou)*np.random.normal(0, 1,size=1).repeat(N_x//2)*np.sqrt(dt)
        x_ou_pi +=dx_pi

        # Store the selected recurrent weights at the original interval.
        if store_weights and t%skip==0:
            track_W[:track_num_n,t//skip]=W[x_idx_e,y_idx_e]
            track_W[track_num_n:,t//skip]=W[x_idx_i,N_E+y_idx_i]

        ext_act = ex_weights @ g_elig_X #np.matmul(ex_weights,convolved_gX)
        rec_act = W @ np.concatenate((g_elig_E, g_elig_I))
        Lam = rec_act+ext_act

        idx = np.where(Lam<0)
        Lam[idx]=0
        # Generate spikes
        random_numE = np.random.rand(N_E)
        random_numI = np.random.rand(N_I)
        random_numX = np.random.rand(N_x)

        if t < bins_store:
            if store_rates:
                spike_trains_E[random_numE < (Lam[:N_E] * dt), t] = time[t]
                spike_trains_I[random_numI < (Lam[N_E:] * dt), t] = time[t]
                spike_trains_X[random_numX < (lambda_X * dt), t] = time[t]
        else:
            if store_rates:
                spike_trains_E[:, :-1] = spike_trains_E[:, 1:]
                spike_trains_I[:, :-1] = spike_trains_I[:, 1:]
                spike_trains_X[:, :-1] = spike_trains_X[:, 1:]
                spike_trains_E[random_numE > (Lam[:N_E] * dt), -1] = 0
                spike_trains_I[random_numI > (Lam[N_E:] * dt), -1] = 0
                spike_trains_X[random_numX > (lambda_X * dt), -1] = 0
                
                spike_trains_E[random_numE < (Lam[:N_E] * dt), -1] = time[t]
                spike_trains_I[random_numI < (Lam[N_E:] * dt), -1] = time[t]
                spike_trains_X[random_numX < (lambda_X * dt), -1] = time[t]
        
        Espike=np.argwhere(random_numE < (Lam[:N_E] * dt)) #np.argwhere(spike_trains_E[:,idx1]>0)
        Ispike=np.argwhere(random_numI < (Lam[N_E:] * dt)) #np.argwhere(spike_trains_I[:,idx1]>0)

        if t%skip==0:
            track_lam_E[:track_num_n,t//skip] = Lam[:track_num_n]

            track_mean_rates[0,t//skip] = np.mean(Lam[:N_E])
            track_mean_rates[1,t//skip] = np.mean(Lam[N_E:])

        # Let the dynamics approach steady state before turning on plasticity.
        start_plastic =4.
        if plastic and time[t]>start_plastic:
            W[Espike[Espike<=N_E],:N_E]+=np.tile(tau_rprim*eta_ee*B12*x[0,:N_E].transpose(),(np.count_nonzero(Espike<=N_E),1) )
            W[:N_E,Espike[Espike<=N_E]]+=np.tile((tau_rprim*eta_ee*B21*(x[0,:N_E]-alpha_ee)),(np.count_nonzero(Espike<=N_E),1)).transpose() #* J[0:N_E,Espike[Espike<=N_E]] 
            W[:N_E,Ispike[Ispike<=N_I]+N_E]-=np.tile(tau_rprim*eta_ei*B21_inh*(x[0,:N_E]-alpha_ei),(np.count_nonzero(Ispike<=N_I),1)).transpose() #* (J[:N_E,Ispike[Ispike<=N_I]+N_E])
            W[Espike[Espike<=N_E],N_E:]-=np.tile(tau_rprim*eta_ei*B12_inh*x[0,N_E:].transpose(),(np.count_nonzero(Espike<N_E),1) )                  
        
        np.fill_diagonal(W,0)
        W[:N_E,:N_E][W[:N_E,:N_E] < 0] = 0
        W[:N_E,N_E:][W[:N_E,N_E:] > 0] = 0
        
        tau_r_I=2*tau_r
        x[0,:N_E]-=dt*x[0,:N_E]/tau_STDP
        x[0,N_E:]-=dt*x[0,N_E:]/tau_STDP
        g_elig_E-=dt*g_elig_E/tau_r
        g_elig_I-=dt*g_elig_I/tau_r_I
        g_elig_X-=dt*g_elig_X/tau_r

        # use the exponential kernel as an eligibility trace like term
        idx =np.where(random_numE < (Lam[:N_E] * dt))
        x[0,idx] += 1  # add spikes to elig trace
        if normalize:
            g_elig_E[idx] += 1 /tau_r #np.sqrt(tau_r)  # add spikes to elig trace
        else:
            g_elig_E[idx] += 1  #np.sqrt(tau_r)  # add spikes to elig trace

        # use the exponential kernel as an eligibility trace like term
        idx =np.where(random_numI < (Lam[N_E:] * dt))
        if normalize:
            g_elig_I[idx] += 1/tau_r_I #np.sqrt(tau_r)  # add spikes to elig trace
        else:
            g_elig_I[idx] += 1 #np.sqrt(tau_r)  # add spikes to elig trace
        idx = N_E+np.array(idx)
        x[0,idx] += 1  # add spikes to elig trace

        # use the exponential kernel as an eligibility trace like term
        idx =np.where(random_numX < (lambda_X * dt))
        if normalize:
            g_elig_X[idx] += 1/tau_r #np.sqrt(tau_r)  # add spikes to elig trace
        else:
            g_elig_X[idx] += 1 #np.sqrt(tau_r)  # add spikes to elig trace

        if t %10000==0:
            print('b',b,'THEORY LAM',tr[0],'poisson lamb',np.mean(Lam[:N_E]))
        if store_theory_lamb and plot_theory:
            rE = tr[0] #exp_lam[0]
            rI = tr[1] #exp_lam[1]
            r_vec = np.array([tr[0], tr[1]])
            rx = np.array([aE,aI])
            
            NW_theory[0,0] = (N_E-1)*tw[0]
            NW_theory[0,1] = (N_I)*tw[1]

            tau_r_vec=np.array([tau_r,tau_r_I])
            if normalize:
                if bernouilli:
                    drdt = ( -r_vec  + p*(NW_theory @ r_vec + Num_neur//2*Wx_theory @ rx))/tau_r_vec
                else:
                    drdt = ( -r_vec +(NW_theory @ r_vec + Num_neur//2*Wx_theory @ rx))/tau_r_vec
            else:
                drdt = ( -r_vec / tau_r_vec + (NW_theory @ r_vec + Num_neur//2*Wx_theory @ rx))

            if plastic and time[t]>start_plastic:
                if time[t]>start_plastic + 2:
                    skip_cov= 5000
                else:
                    skip_cov= 1000
                if t%skip_cov==0:
                    
                    if bernouilli:
                        cov_term = calculate_CSD(Num_neur//2,2,2,p*NW_theory, p*Wx_theory,tau_STDP,tau_r,tau_ou,c_x,aE,aI,sigma_p,sigma_s,rE,rI)
                    else:
                        cov_term = calculate_CSD(Num_neur//2,2,2,NW_theory, Wx_theory,tau_STDP,tau_r,tau_ou,c_x,aE,aI,sigma_p,sigma_s,rE,rI)
                    print('cov',cov_term)
                if normalize:
                    dwdt[0,0] = (tau_rprim / tau_wee) * (tau_STDP*rE*(rE-b)+ cov_term[0,0])
                    dwdt[0, 1] = -( tau_rprim / tau_wei) * (tau_STDP*rI*(rE-b)+cov_term[0,1])
                else:
                    dwdt[0,0] = (1. / tau_wee) * (tau_STDP*rE*(rE-b)+ cov_term[0,0])
                    dwdt[0, 1] = -( 1 / tau_wei) * (tau_STDP*rI*(rE-b)+cov_term[0,1])
                
            tw = tw + dwdt[0,:] * dt
            tr = tr + drdt * dt 
            
            if t%10000==0:
                print('theory_weights[1,t]',tw[1],'theory_weights[0,t]',tw[0])
                print('##############################################################')
                print('##############################################################')
                print('##############################################################')
                print('theory_rates[:, t]',tr[0],'rE',rE)
            if t%skip==0:
                theory_weights[:, t//skip] = tw
                theory_rates[:, t//skip] = tr#[rE,rI] #[rE,rI] #theory_rates[:, t] + drdt * dt
            
        if store_mean_weights and t%skip==0:
            ee_weights=W[:N_E,:N_E]
            ei_weights=W[:N_E,N_E:]
            nonzero_elements_ee = ee_weights[ee_weights != 0]
            nonzero_elements_ei = ei_weights[ei_weights != 0]
            #    Calculate the mean of the nonzero elements
            mean_nonzero_ee = np.mean(nonzero_elements_ee)
            mean_nonzero_ei = np.mean(nonzero_elements_ei)

            if bernouilli:
                track_mean_W[0,t//skip]=mean_nonzero_ee
                track_mean_W[1,t//skip]=mean_nonzero_ei
            else:
                track_mean_W[0,t//skip]=mean_nonzero_ee
                track_mean_W[1,t//skip]=mean_nonzero_ei
            if t%8000==0:
                print('track_mean_W_ee[t]',mean_nonzero_ee)#,'track_std_W_ee[t]',track_std_W_ee[t])
                print('track_mean_W_ei[t]',mean_nonzero_ei)#,'track_std_W_ei[t]',track_std_W_ei[t])

    time_reduced = time[::skip]
    
    
    # Optional diagnostic plotting retained exactly as in the uploaded file.
    if axs is not None:
        axs[0,0].plot(time_reduced,track_W[:track_num_n,:-1].T,color='tab:grey',linewidth=.5)
        if plot_theory:
            axs[0,0].plot(time_reduced,theory_weights[0,:-1],color='k',label='theory',linewidth=.5)
            axs[1,0].plot(time_reduced,theory_weights[0,:-1],color='k',label='theory',linewidth=.5)
        print('track_mean_W[:track_num_n,:-1]',track_mean_W[0,:10])
        axs[0,0].plot(time_reduced,track_mean_W[0,:-1],color='tab:red',linestyle='solid',label='sim ee',linewidth=.5)
        axs[1,0].plot(time_reduced,track_mean_W[0,:-1],color='tab:red',linestyle='solid',label='sim ee',linewidth=.5)

        axs[0,0].plot(time_reduced,track_W[track_num_n:,:-1].T,color='tab:grey',linewidth=.5)
        if plot_theory:
            axs[0,0].plot(time_reduced,theory_weights[1,:-1],color='k',linewidth=1.,label='theory')
            axs[1,1].plot(time_reduced,theory_weights[1,:-1],color='k',linewidth=1.,label='theory')
        axs[0,0].plot(time_reduced,track_mean_W[1,:-1],color='tab:blue',linestyle='solid',linewidth=1.,label='sim ei')
        axs[1,1].plot(time_reduced,track_mean_W[1,:-1],color='tab:blue',linestyle='solid',linewidth=1.,label='sim ei')

        axs[0,1].plot(time_reduced,track_lam_E[:,:-1].T,color='tab:grey',linewidth=1.)
        axs[0,1].plot(time_reduced,track_mean_rates[0,:-1],color='red',label='mean ')
        if plot_theory:
            axs[0,1].plot(time_reduced,theory_rates[0,:-1],label='theory mean',linewidth=1.,color='k')
        axs[0,1].set_ylim(0,35)
        axs[0,0].set_ylim(-.0025,.0025)
        axs[0,2].set_ylim(0,55)
        for j in range(3):
            axs[0,j].spines['top'].set_visible(False)
            axs[0,j].spines['right'].set_visible(False)
            axs[0,j].spines['bottom'].set_linewidth(1)
        
    if plot_theory:
        theory_lamb = np.vstack([theory_rates[0,:],theory_rates[1,:]])
    else:
        theory_weights = np.array([[0,0],[0,0]])
        theory_lamb = np.array([[0,0],[0,0]])

    track_W_mean = np.vstack([track_mean_W[0,:],track_mean_W[1,:]])

    return time ,track_W,track_W_mean, track_mean_rates,time_reduced,track_lam_E,track_lam_I, spike_trains_E, spike_trains_I,theory_weights[0,:],theory_weights[1,:],theory_lamb,balance_index, b


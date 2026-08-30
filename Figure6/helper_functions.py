"""Shared covariance and connectivity helpers for the Figure 6 simulations.

Function signatures and numerical operations are unchanged.
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Fourier-domain covariance calculations
def OU_covariance_matrix(Sim_N, N_x,a,b, sigma_s,sigma_p,tau_ou,c,omega):
    cov_matrix = (c*sigma_s**2 ) * np.ones((N_x, N_x))
    extra_diagonal = ((1 - c) * sigma_p**2)

    cov_matrix[:N_x//2,:N_x//2] += extra_diagonal
    cov_matrix[N_x//2:,N_x//2:] += extra_diagonal

    cov_matrix = cov_matrix[:, :, np.newaxis]
    cov_matrix = cov_matrix*(1/(1+(2*np.pi*tau_ou*omega)**2))

    cov_matrix[np.arange(N_x//2), np.arange(N_x//2)] += a/Sim_N
    cov_matrix[N_x//2+np.arange(N_x//2), N_x//2+np.arange(N_x//2)] += b/Sim_N
    cov_matrix=Sim_N*cov_matrix*Sim_N

    return cov_matrix

def calculate_CSD(Sim_N,N,N_x,W, W_x,tau_STDP, tau_r,tau_ou,c,a,b,sigma_p,sigma_s,lamb_e,lamb_i,normalize=True):
    # Integrate the cross spectrum over the frequency grid
    omega=np.linspace(-2500,2500,100000)
    domega=omega[1]-omega[0]
    omega = omega[np.newaxis, np.newaxis, :]
    lambda_cov = OU_covariance_matrix(Sim_N,N_x,a,b,sigma_s,sigma_p,tau_ou,c,omega)

    D = np.zeros(shape=(N,N))
    D[np.arange(N//2), np.arange(N//2)] = lamb_e
    D[N//2+np.arange(N//2), N//2+np.arange(N//2)]=lamb_i

    Id = np.eye(N)
    Id = np.expand_dims(Id,axis=2)
    Id=np.repeat(Id,len(omega[0,0,:]),axis=2)
    D = np.expand_dims(D,axis=2)
    D=np.repeat(D,len(omega[0,0,:]),axis=2)

    W_xT=np.expand_dims(W_x.T,axis=2)
    W_xT=np.repeat(W_xT,len(omega[0,0,:]),axis=2)
    WT=np.expand_dims(W.T,axis=2)
    WT=np.repeat(WT,len(omega[0,0,:]),axis=2)

    W_x=np.expand_dims(W_x,axis=2)
    W_x=np.repeat(W_x,len(omega[0,0,:]),axis=2)

    W=np.expand_dims(W,axis=2)
    W=np.repeat(W,len(omega[0,0,:]),axis=2)
    Full_W = W.astype(np.complex128, copy=True)
    Full_WT = WT.astype(np.complex128, copy=True)

    tau_ri = 2*tau_r
    if normalize:
        exp_term_tau_rE1 = 1 /(1+1j*2*np.pi*tau_r*omega)
        exp_term_tau_rE2 = 1/(1-1j*2*np.pi*tau_r*omega)

        exp_term_tau_rI1 = 1 /(1+1j*2*np.pi*tau_ri*omega)
        exp_term_tau_rI2 = 1/(1-1j*2*np.pi*tau_ri*omega)

        exp_term_tau_r_vec_1 = np.array([exp_term_tau_rE1,exp_term_tau_rI1])
        exp_term_tau_r_vec_2 = np.array([exp_term_tau_rE2,exp_term_tau_rI2])
    else:
        exp_term_tau_rE1 = tau_r /(1+1j*2*np.pi*tau_r*omega)
        exp_term_tau_rE2 = tau_r /(1-1j*2*np.pi*tau_r*omega)

    exp_term_tau_stdp = tau_STDP /(1+1j*2*np.pi*tau_STDP*omega)

    left_wx_term = W_x*exp_term_tau_rE1
    right_wx_term = (W_xT)*exp_term_tau_rE2

    right_Xcov = np.einsum('ijk,jlk->ilk', lambda_cov, left_wx_term)
    X_cov = np.einsum('ijk,jlk->ilk', right_wx_term, right_Xcov)

    Full_W[:,0,:] =  W[:,0,:]*exp_term_tau_r_vec_1[0,0,0,:]
    Full_W[:,1,:] =  W[:,1,:]*exp_term_tau_r_vec_1[1,0,0,:]
    left_inverted = np.transpose(Id - Full_W, (2, 0, 1))
    Full_WT[0,:,:] =  WT[0,:,:]*exp_term_tau_r_vec_2[0,0,0,:]
    Full_WT[1,:,:] =  WT[1,:,:]*exp_term_tau_r_vec_2[1,0,0,:]
    right_inverted = np.transpose(Id - Full_WT, (2, 0, 1))
    inv_term_1 = np.linalg.inv(left_inverted)
    inv_term_2 = np.linalg.inv(right_inverted)

    inv_term_1 = np.transpose(inv_term_1, (1,2,0))
    inv_term_2 = np.transpose(inv_term_2, (1,2,0))

    right_Ktilde = np.einsum('ijk,jlk->ilk', D/(Sim_N-1)+X_cov, inv_term_2)
    Ktilde_SSdf =exp_term_tau_stdp*np.einsum('ijk,jlk->ilk', inv_term_1, right_Ktilde)
    Ktilde_SSdf = np.sum(Ktilde_SSdf,axis=2)*domega

    del Full_WT
    del Full_W
    del W
    return Ktilde_SSdf


# ---------------------------------------------------------------------------
# Connectivity construction
def generate_bernouilli_wx(init_weights, N, N_X,num_per_pool):
    p = 0.1

    weights = np.zeros(shape=(N,N_X//num_per_pool))
    num = N_X//num_per_pool
    weights[:N//2, :num//2] = (np.random.rand(N//2, num//2) < p) * init_weights[0]
    weights[:N//2, num//2:] = 0
    weights[N//2:, :num//2] = 0
    weights[N//2:, num//2:] = (np.random.rand(num//2, N//2) < p) * init_weights[1]

    return weights, p

def generate_bernouilli_weight_array(init_weights, N_E, N_I):
    p = 0.1

    weights = np.zeros(shape=((N_E+N_I), (N_E+N_I)))
    w_EE = init_weights[0,0]
    weights[:N_E, :N_E] = (np.random.rand(N_E, N_E) < p) * w_EE
    w_EI = init_weights[0,1]
    weights[:N_E, N_E:] = (np.random.rand(N_E, N_I) < p) * w_EI
    w_IE = init_weights[1,0]
    weights[N_E:, :N_E] = (np.random.rand(N_I, N_E) < p) * w_IE
    w_II = init_weights[1,1]
    weights[:N_I, :N_I] = (np.random.rand(N_I, N_I) < p) * w_II

    return weights


# ---------------------------------------------------------------------------
# Spike-count and rate utilities
def calculate_spike_count_covariance(spike_trains1,spike_trains2, window_size_ms, dt):
    """
    Calculate the average spike count covariance between neurons.

    Parameters:
    spike_trains (np.ndarray): An N x T binary array of spike trains (N neurons, T time points).
    window_size_ms (int): The size of the time window in milliseconds.
    tau_ou (float): The time step in milliseconds.

    Returns:
    np.ndarray: An N x N array of spike count covariances.
    """
    N_1, T = spike_trains1.shape
    N_2, T = spike_trains2.shape

    # Number of time points in each window
    window_size = int(window_size_ms / dt)

    # Number of windows
    transient = 3*T//4
    num_windows = (T-transient) // window_size

    # Bin spikes and collect per-window spike counts
    print('num_windows',num_windows)
    spike_counts1 = np.zeros((N_1, num_windows))
    spike_counts2 = np.zeros((N_2, num_windows))


    for k in range(num_windows):
        binned_trains1 = bin_spike_trains(spike_trains1[:,transient:], dt, T)
        binned_trains2 = bin_spike_trains(spike_trains2[:,transient:], dt, T)
        start_idx = k * window_size
        end_idx = start_idx + window_size
        spike_counts1[:, k] = np.sum(binned_trains1[:, start_idx:end_idx], axis=1)
        spike_counts2[:, k] = np.sum(binned_trains2[:, start_idx:end_idx], axis=1)

    # Compute per-neuron means and the pairwise covariance matrix
    mean_spike_counts1 = np.mean(spike_counts1, axis=1)
    mean_spike_counts2 = np.mean(spike_counts2, axis=1)

    covariance_matrix = np.zeros((N_1, N_2))

    for i in range(N_1):
        for j in range(N_2):
            covariance_matrix[i, j] = np.mean((spike_counts1[i, :] - mean_spike_counts1[i]) * (spike_counts2[j, :] - mean_spike_counts2[j]))

    print('covariance_matrix',covariance_matrix)
    return covariance_matrix

def generate_sep_poisson_neurons(base_ex,N_X, time_steps, dt, sigma_s, sigma_p, c_x):
    # Draw shared and private white-noise processes
    xi_s = np.random.normal(0, 1, size=len(time_steps))
    xi_p = np.random.normal(0, 1, size=(N_X, len(time_steps)))
    # Form the external-rate process
    lambda_xx = base_ex + c_x * sigma_s * xi_s
    return lambda_xx

def generate_weight_array(init_weights, num_E, num_I):
    variances = [.0000005,.0000005,.0000005,.0000005]


    row_num = [num_E,num_I]
    Num_neur=num_E+num_I
    rep_weights = np.zeros(shape=((num_E+num_I), (num_E+num_I)))
    shapes = [(num_E, num_E),(num_E, num_I),(num_I, num_E),(num_I, num_I)]
    ct=0

    for i in range(init_weights.shape[0]):
        for j in range(init_weights.shape[1]):
            variance=variances[ct]
            mean_value = init_weights[i, j]
            values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=shapes[ct])
            idx = np.where(values<0)
            values[idx]=0
            rep_weights[i * num_E:(i + 1) * num_E, j * num_E:(j + 1) * num_E] = values
            ct +=1
    plot=False
    if plot:
        plt.hist(rep_weights[:num_E,:num_E].flatten(),bins=50,alpha=.5,label='wee')
        plt.hist(rep_weights[:num_E,num_E:].flatten(),bins=50,alpha=.5,label='wei')
        plt.hist(rep_weights[num_E:,num_E:].flatten(),bins=50,alpha=.5,label='wii')
        plt.hist(rep_weights[num_E:,:num_E].flatten(),bins=50,alpha=.5,label='wie')
        plt.legend()
        plt.show()
    return rep_weights


def generate_diagonal_wx(init_weights, N, N_X):
    variances = [.0000005,.0000005]

    repeated_weights = np.zeros(shape=(N,N_X))
    shapes = N//2
    ct=0

    for i in range(init_weights.shape[0]):
        variance=variances[i]
        mean_value = init_weights[i]
        values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=(shapes))
        repeated_weights[np.arange(N//2), np.arange(N//2)+N//2] = values
    plot=False
    if plot:
        plt.hist(repeated_weights.flatten(),alpha=.5)
        plt.show()
    return repeated_weights

def generate_block_wx(init_weights, N, N_X):
    variances = [.0000001,.0000001]

    repeated_weights = np.zeros(shape=(N,N_X))
    shapes = N//2
    ct=0

    for i in range(init_weights.shape[0]):
        variance=variances[i]
        mean_value = init_weights[i]
        values = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=(shapes,shapes))
        repeated_weights[i * N//2:(i + 1) * N//2, i * N//2:(i + 1) * N//2] = values

    return repeated_weights


def calculate_firing_rate(N,spike_times, window_size, time_step,curr_t):
    firing_rate = np.zeros((N))

    for i in range(N):
        # Count spikes in the requested window
        start = int( curr_t - window_size )
        end = int(curr_t)
        spike_count = np.count_nonzero(spike_times[i, start:end])
        firing_rate[i] = spike_count / (window_size*time_step)

    return firing_rate

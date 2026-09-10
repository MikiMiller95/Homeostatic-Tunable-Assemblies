"""Deterministic two-population rate simulation used by Figure 5."""
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Deterministic pulse-response simulation
# ──────────────────────────────────────────────────────────────────────────


def run_sim(N, w_EE, w_IE, w_II, w_EI, tau_ou, tau_STDP, tau_rprim, tau_r, tau_w_e, tau_w_i, w_EX, w_IX, aE, aI, b, c_x, sigma_s, sigma_p, timesteps, pulse_num=1):
    """Simulate the deterministic E-I rate dynamics with the original pulse protocol."""
    # Simulation grid and recorded state.
    ms_per_sec = 1000.0
    dt = 0.1 / ms_per_sec
    time = np.linspace(0, timesteps, int(timesteps / dt))
    dwdt = np.zeros((2, 2, len(time)))
    rec_weights = np.zeros((2, 2, len(time)))
    rates = np.zeros((2, len(time)))
    pulse_trace = np.zeros_like(time)
    # Fixed external-input matrix and initial recurrent weights.
    Wx_sim = np.zeros((2, 2))
    Wx_sim[0, 0] = w_EX
    Wx_sim[1, 1] = w_IX
    rec_weights[:, :, 0] = np.array([[w_EE, -w_EI], [w_IE, -w_II]])
    # Configure either the single perturbation or the repeated pulse train.
    if pulse_num == 1:
        pert_time = 6.8
        pulse_amp = 0.02 / tau_r
    else:
        pulse_amp = 0.003 / tau_r
        pulse_width = 0.03
        burst_duration = 5.0
        silent_duration = 5.0
        pulse_interval = 0.1
        pulse_start = 15.0
        cycle_duration = burst_duration + silent_duration
        pulse_times = []
        t = pulse_start
        while t < timesteps:
            burst_end = t + burst_duration
            burst_pulses = np.arange(t, min(burst_end, timesteps), pulse_interval)
            pulse_times.extend(burst_pulses)
            t += cycle_duration
        width_steps = int(np.round(pulse_width / dt))
        burst_mask = np.zeros_like(time, dtype=bool)
        for pt in pulse_times:
            i0 = int(np.round(pt / dt))
            burst_mask[i0:i0 + width_steps] = True
    # Evolve presynaptic traces; rates remain instantaneous firing rates.
    r_vec = np.zeros(2)
    rx = np.zeros(2)
    tau_r_vec = np.array([tau_r, tau_r * 2.0])
    for i in range(len(time) - 1):
        if i % 100000 == 0:
            print(time[i])
        W = rec_weights[:, :, i]
        # Apply each pulse to the E source trace before reconstructing rates.
        if pulse_num == 1:
            if abs(time[i] - pert_time) < 10 * dt:
                r_vec[0] += pulse_amp
                pulse_trace[i] = pulse_amp
        elif burst_mask[i]:
            r_vec[0] += pulse_amp
            pulse_trace[i] = pulse_amp
        rates[:, i] = np.maximum(N * (W @ r_vec + Wx_sim @ rx), 0.0)
        drdt = (rates[:, i] - r_vec) / tau_r_vec
        r_vec = r_vec + drdt * dt
        rx = rx + dt * (np.array([aE, aI]) * (time[i] > 0.0) - rx) / tau_r
        rec_weights[:, :, i + 1] = rec_weights[:, :, i] + dwdt[:, :, i] * dt
        rates[:, i + 1] = np.maximum(N * (rec_weights[:, :, i + 1] @ r_vec + Wx_sim @ rx), 0.0)
    return (rates, time, np.abs(rec_weights), pulse_trace, pulse_amp)

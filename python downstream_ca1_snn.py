import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SNN Simulation Settings
# ==========================================
dt = 0.1          # Time step (ms)
T_sim = 1000.0    # Total simulation duration (ms)
time = np.arange(0, T_sim + dt, dt)
steps = len(time)

# ==========================================
# 2. Network Architecture & Parameters (Izhikevich SNN)
# ==========================================
N = 1000          # Total number of neurons
N_e = 800         # Excitatory (Pyramidal) principal cells
N_i = 200         # Inhibitory (Fast-Spiking) interneurons

# Initialize Izhikevich parameters arrays
a = np.zeros(N)
b = np.zeros(N)
c = np.zeros(N)
d = np.zeros(N)

# Pyramidal (Excitatory - Regular Spiking / RS)
a[:N_e] = 0.02
b[:N_e] = 0.2
c[:N_e] = -65.0
d[:N_e] = 8.0

# Inhibitory Interneurons (Fast Spiking / FS)
a[N_e:] = 0.1
b[N_e:] = 0.2
c[N_e:] = -65.0
d[N_e:] = 2.0

# Reversal potentials for synaptic currents
V_AMPA = 0.0      # mV
V_GABA = -70.0    # mV

# Susceptibility coefficient to orexin (higher for pyramidal than interneurons)
eta = np.zeros(N)
eta[:N_e] = 0.05   # Pyramidal neurons
eta[N_e:] = 0.02   # Inhibitory interneurons

# ==========================================
# 3. State Variables & Connectivity Initialization
# ==========================================
v = -65.0 * np.ones(N)  # Membrane potential (mV)
u = b * v               # Membrane recovery variable

# Random sparse directional connectivity matrix scaled for stability
np.random.seed(42)
W = np.random.randn(N, N) * 0.5
W = np.maximum(W, 0)    # Non-negative synaptic weights
W = W / np.sqrt(N)      # Scaling factor for network stability

sigma_noise = 0.15      # Background Gaussian noise standard deviation

# Storage arrays for LFP and spike times
lfp_history = np.zeros(steps)
spike_times = []

# Orexin effective drive level (e.g., I_O = 2.0 for Control Phase, 0.44 for Late Failure Phase)
I_O_val = 2.0 

# ==========================================
# 4. Main SNN Simulation Loop (Euler Integration)
# ==========================================
for k in range(steps):
    t = time[k]
    
    # External baseline drive modulated dynamically by orexin concentration (I_O)
    I_base = np.random.normal(5.0, 1.0, N)
    I_ext = I_base * (1.0 + eta * I_O_val) + np.random.normal(0, sigma_noise, N)
    
    # Compute total synaptic currents for each neuron
    I_syn = np.zeros(N)
    for i in range(N):
        # Excitatory and inhibitory synaptic inputs
        exc_input = np.sum(W[i, :N_e] * (v[i] - V_AMPA))
        inh_input = np.sum(W[i, N_e:] * (v[i] - V_GABA))
        I_syn[i] = -exc_input - inh_input

    # Izhikevich neuronal membrane potential differential equations
    dv = 0.04 * (v**2) + 5.0 * v + 140.0 - u + I_syn + I_ext
    du = a * (b * v - u)
    
    v += dv * dt
    u += du * dt
    
    # Spike detection and auxiliary resetting condition (v >= 30 mV)
    fired = np.where(v >= 30.0)[0]
    if len(fired) > 0:
        v[fired] = c[fired]
        u[fired] += d[fired]
        for idx in fired:
            spike_times.append((t, idx))
            
    # Compute Local Field Potential (LFP) approximation by averaging pyramidal membrane potentials
    lfp_history[k] = np.mean(v[:N_e])

# ==========================================
# 5. Visualization (Raster Plot & LFP)
# ==========================================
plt.figure(figsize=(12, 8))

# Raster Plot
plt.subplot(2, 1, 1)
if len(spike_times) > 0:
    t_spikes, id_spikes = zip(*spike_times)
    plt.scatter(t_spikes, id_spikes, s=0.5, c='b', marker='.')
plt.title(f'CA1 Microcircuit Spiking Activity (Orexin Drive $I_O$ = {I_O_val})')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID (0-799: Excitatory, 800-999: Inhibitory)')
plt.xlim(0, T_sim)
plt.ylim(0, N)
plt.grid(True)

# Local Field Potential (LFP) Proxy
plt.subplot(2, 1, 2)
plt.plot(time, lfp_history, 'k-', linewidth=0.8)
plt.title('Approximated Local Field Potential (LFP)')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.xlim(0, T_sim)
plt.grid(True)

plt.tight_layout()
plt.show()
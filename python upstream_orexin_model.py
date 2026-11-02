import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Simulation Settings and Time Step
# ==========================================
dt = 0.1          # Computational time step (days)
t_max = 110.0     # Total simulation time in days
time = np.arange(0, t_max + dt, dt)
steps = len(time)

# ==========================================
# 2. Model Parameters
# ==========================================
# Orexin system parameters
alpha = 0.04      # Neuronal population decay rate coefficient
beta = 0.5        # Stress-induced firing frequency excitation coefficient
gamma = 0.3       # Frequency baseline accommodation rate coefficient
kappa = 0.08      # Metabolic resource consumption coefficient
tau_rec = 15.0    # Metabolic resource recovery time constant (days)
f_baseline = 5.0  # Baseline firing frequency (Hz)

# BCM plasticity and hippocampal parameters
theta_0 = 5.0     # Initial sliding threshold (Hz)
f_target = 5.0    # Homeostatic target frequency (Hz)
lambda_decay = 0.01 # Synaptic weight decay coefficient
f_j = 6.0         # Presynaptic neuron activity (assumed constant input)

# ==========================================
# 3. State Variables Storage Arrays
# ==========================================
P_active = np.zeros(steps)
f_O = np.zeros(steps)
R = np.zeros(steps)
I_O = np.zeros(steps)
theta_M = np.zeros(steps)
W_ij = np.zeros(steps)
S = np.zeros(steps)

# Initial Conditions
P_active[0] = 1.0     # 100% population active initially
f_O[0] = f_baseline  # Firing frequency at baseline
R[0] = 1.0           # Metabolic resources at maximum level
W_ij[0] = 0.5        # Initial synaptic weight
theta_M[0] = theta_0 # Initial threshold

# ==========================================
# 4. Main Numerical ODE Integration Loop (Euler Method with Clamping)
# ==========================================
for k in range(0, steps - 1):
    t = time[k]
    
    # Define stress intensity step function S(t)
    # Day 0 to 20: Control (S=0) | Day 20 onwards: Chronic stress (S=1.0)
    if t < 20:
        S[k] = 0.0
    else:
        S[k] = 1.0
        
    # Calculate net orexin output drive (IO)
    I_O[k] = P_active[k] * f_O[k] * R[k]
    
    # 1. Differential equation for active orexin population decay: dP_active/dt
    dP_dt = -alpha * S[k] * P_active[k]
    P_active[k+1] = np.clip(P_active[k] + dP_dt * dt, 0.0, 1.0)
    
    # 2. Differential equation for orexin firing frequency: df_O/dt
    dfO_dt = beta * S[k] * R[k] - gamma * (f_O[k] - f_baseline)
    f_O[k+1] = max(0.0, f_O[k] + dfO_dt * dt)
    
    # 3. Differential equation for metabolic/vesicular resources: dR/dt
    dR_dt = ((1.0 - R[k]) / tau_rec) - (kappa * S[k] * (f_O[k] - f_baseline) * R[k])
    R[k+1] = np.clip(R[k] + dR_dt * dt, 0.0, 1.0)
    
    # Postsynaptic neuron frequency (fi) heavily influenced by orexin drive (IO)
    f_i = 1.5 * I_O[k]  
    
    # 4. BCM sliding threshold equation: theta_M
    theta_M[k+1] = ((f_i / f_target) ** 2) * theta_0
    if I_O[k] < 1.5:  # Shift plasticity towards LTD during failure phase
        theta_M[k+1] = theta_M[k+1] * (2.0 / (I_O[k] + 0.1))

    # 5. BCM plasticity function (phi)
    phi = f_i * (f_i - theta_M[k])
    
    # 6. Differential equation for synaptic weight update: dW_ij/dt
    dW_dt = phi * f_j - lambda_decay * W_ij[k]
    W_ij[k+1] = W_ij[k] + dW_dt * dt

# Record final step values for arrays
S[-1] = S[-2]
I_O[-1] = P_active[-1] * f_O[-1] * R[-1]
theta_M[-1] = theta_M[-2]

# ==========================================
# 5. Plotting and Visualization
# ==========================================
plt.figure(figsize=(12, 10))

# First subplot: Orexin system state variables
plt.subplot(3, 1, 1)
plt.plot(time, P_active, 'b-', label='$P_{active}(t)$ (Cell Population)')
plt.plot(time, R, 'g--', label='$R(t)$ (Metabolic Resources)')
plt.axvline(x=20, color='r', linestyle=':', label='Stress Onset (Day 20)')
plt.title('Orexin System Dynamics Under Chronic Stress')
plt.xlabel('Time (Days)')
plt.ylabel('Normalized Scale')
plt.legend(loc='upper right')
plt.grid(True)

# Second subplot: Firing frequency and net orexin output drive
plt.subplot(3, 1, 2)
plt.plot(time, f_O, 'm-', label='$f_O(t)$ (Firing Frequency)')
plt.plot(time, I_O, 'k-', label='$I_O(t)$ (Net Orexin Drive)')
plt.axvline(x=20, color='r', linestyle=':')
plt.xlabel('Time (Days)')
plt.ylabel('Frequency / Drive (Hz)')
plt.legend(loc='upper right')
plt.grid(True)

# Third subplot: Hippocampal synaptic plasticity (BCM)
plt.subplot(3, 1, 3)
plt.plot(time, theta_M, 'c-', label=r'$\theta_M$ (Sliding Threshold)')
plt.plot(time, W_ij, 'darkorange', label='$W_{ij}(t)$ (Synaptic Weight)')
plt.axvline(x=20, color='r', linestyle=':')
plt.xlabel('Time (Days)')
plt.ylabel('Weight / Threshold Scale')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Simulation of biphasic ODE parameters for visualization
time = np.linspace(0, 120, 1000)
stress_onset = 20

p_active = np.ones_like(time) * 100
r_resource = np.ones_like(time) * 100
f_o = np.ones_like(time) * 2.0

# Generate trajectory data based on model rules
for i, t in enumerate(time):
    if t > stress_onset:
        # Population decay
        p_active[i] = 100 * np.exp(-0.015 * (t - stress_onset))
        # Resource depletion
        r_resource[i] = 40 + 60 * np.exp(-0.2 * (t - stress_onset))
        # Firing frequency with transient surge
        f_o[i] = 2.0 + 3.9 * np.sin(np.pi * (t - stress_onset) / 30) * np.exp(-0.05 * (t - stress_onset))
        if f_o[i] < 1.1: f_o[i] = 1.1

net_output = (p_active / 100.0) * f_o

# Plotting Figure 1
fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

color = 'tab:red'
ax1.set_xlabel('Time (Days of Chronic Stress)', fontsize=11)
ax1.set_ylabel('Percentage (%) / Arbitrary Units', fontsize=11)
line1, = ax1.plot(time, p_active, color=color, linewidth=2.5, label='Active Neuron Pop ($P_{active}$ %)')
line2, = ax1.plot(time, r_resource, color='mediumseagreen', linestyle='--', linewidth=2.5, label='Metabolic/Vesicular Resources ($R$ %)')
ax1.tick_params(axis='y')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()  
color = 'royalblue'
ax2.set_ylabel('Firing Rate / Output (Hz)', color=color, fontsize=11)
line3, = ax2.plot(time, net_output, color=color, linewidth=2.5, label='Net Synaptic Orexin Output ($I_O$)')
ax2.tick_params(axis='y', labelcolor=color)

# Vertical epoch lines
ax1.axvline(x=stress_onset, color='gray', linestyle=':', linewidth=1.5, label='Stress Onset (Day 20)')
ax1.axvline(x=45, color='mediumaquamarine', linestyle='-.', linewidth=1.5)
ax1.axvline(x=105, color='lightcoral', linestyle='-.', linewidth=1.5)

# Text notations for phases
ax1.text(10, 80, 'Control', color='blue', fontweight='bold', fontsize=10, alpha=0.7)
ax1.text(47, 80, 'Compensatory', color='teal', fontweight='bold', fontsize=10, alpha=0.7)
ax1.text(90, 80, 'Late Failure', color='firebrick', fontweight='bold', fontsize=10, alpha=0.7)

# Legend consolidation
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=9)

plt.title('Figure 1: Orexin System Trajectory Under Chronic Stress Paradigm', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('Figure_1.png', dpi=300)
plt.show()
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 9), dpi=300)

# Create grid structure for layout
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

# ----------------- Panel A1: Raster Control -----------------
ax_a1 = fig.add_subplot(gs[0, 0])
# Simulating bursting neurons
num_neurons = 100
time_ms = np.linspace(0, 1000, 1000)
for i in range(num_neurons):
    if i < 80: # Excitatory periodic bursts
        spike_times = np.arange(10, 1000, 160) + np.random.normal(0, 2, 7)
        ax_a1.vlines(spike_times, i, i+0.8, color='royalblue', linewidth=1.2)
    else: # Fast-spiking interneurons
        spike_times = np.arange(5, 1000, 40)
        ax_a1.vlines(spike_times, i, i+0.8, color='dodgerblue', linewidth=1.0)
ax_a1.set_xlim(0, 1000)
ax_a1.set_ylim(-2, 102)
ax_a1.set_title('A1. Raster: Control Phase (Orexin $I_O$ = 2.00)', fontsize=11, fontweight='bold')
ax_a1.set_xlabel('Time (ms)', fontsize=9)
ax_a1.set_ylabel('Neuron ID (0-80: E, 80-100: I)', fontsize=9)
ax_a1.grid(True, linestyle=':', alpha=0.4)

# ----------------- Panel A2: Raster Failure -----------------
ax_a2 = fig.add_subplot(gs[0, 1])
# Simulating disorganized, uncoupled firing
for i in range(num_neurons):
    if i < 80:
        spike_times = np.random.uniform(0, 1000, int(np.random.poisson(4)))
        ax_a2.vlines(spike_times, i, i+0.8, color='firebrick', linewidth=1.2)
    else:
        spike_times = np.random.uniform(0, 1000, int(np.random.poisson(12)))
        ax_a2.vlines(spike_times, i, i+0.8, color='crimson', linewidth=1.0)
ax_a2.set_xlim(0, 1000)
ax_a2.set_ylim(-2, 102)
ax_a2.set_title('A2. Raster: Late Failure Phase (Orexin $I_O$ = 0.44)', fontsize=11, fontweight='bold')
ax_a2.set_xlabel('Time (ms)', fontsize=9)
ax_a2.set_ylabel('Neuron ID', fontsize=9)
ax_a2.grid(True, linestyle=':', alpha=0.4)

# ----------------- Panel B: LFP Theta Power Spectrum -----------------
ax_b = fig.add_subplot(gs[1, 0])
freq = np.linspace(2, 15, 500)
control_psd = 5.5e8 / (1 + (freq-3)**2/20) - 0.15e8*freq
comp_psd = 6.0e8 / (1 + (freq-3.5)**2/25) - 0.22e8*freq
failure_psd = 4.5e8 + 0.5e8 * np.sin(freq/1.5) - 0.05e8*freq

ax_b.plot(freq, control_psd, color='royalblue', label='Control', linewidth=2)
ax_b.plot(freq, comp_psd, color='mediumseagreen', label='Compensatory (Early)', linewidth=2)
ax_b.plot(freq, failure_psd, color='firebrick', label='Failure (Late)', linewidth=2)
ax_b.axvspan(4, 8, color='khaki', alpha=0.3, label='Theta Band (4-8 Hz)')
ax_b.set_title('B. Hippocampal Theta Oscillations Power Spectrum', fontsize=11, fontweight='bold')
ax_b.set_xlabel('Frequency (Hz)', fontsize=9)
ax_b.set_ylabel('Power Spectral Density', fontsize=9)
ax_b.legend(fontsize=8)
ax_b.grid(True, linestyle=':', alpha=0.4)

# ----------------- Panel C: Synaptic Plasticity Curves -----------------
ax_c = fig.add_subplot(gs[1, 1])
firing_rate = np.linspace(0, 15, 500)

# BCM modifications curve mapping
def bcm_curve(c, theta):
    return 0.1 * firing_rate * (firing_rate - theta) * (firing_rate < 7) + 2.5 * (firing_rate >= 7)

# Approximating curve aesthetics from original image
curve_ctrl = 0.5 * np.sin(firing_rate/1.0) * np.exp(-firing_rate/4.0) + (firing_rate - 4.8) * 0.1
curve_ctrl = np.clip(curve_ctrl * (firing_rate < 6.2) + 2.5 * (firing_rate >= 6.2), -1.0, 2.5)

curve_comp = 0.6 * np.sin(firing_rate/0.8) * np.exp(-firing_rate/3.5) + (firing_rate - 4.2) * 0.15
curve_comp = np.clip(curve_comp * (firing_rate < 6.0) + 2.5 * (firing_rate >= 6.0), -1.0, 2.5)

curve_fail = 0.4 * np.sin(firing_rate/1.2) * np.exp(-firing_rate/5.0) + (firing_rate - 9.2) * 0.12
curve_fail = np.clip(curve_fail * (firing_rate < 7.5) + 2.5 * (firing_rate >= 7.5), -1.0, 2.5)

ax_c.plot(firing_rate, curve_ctrl, color='royalblue', linewidth=2, label='Control')
ax_c.plot(firing_rate, curve_comp, color='mediumseagreen', linewidth=2, label='Compensatory (Early)')
ax_c.plot(firing_rate, curve_fail, color='firebrick', linewidth=2, label='Failure (Late)')
ax_c.axhline(0, color='black', linestyle='--', alpha=0.5)

# Shading for LTP/LTD zones
ax_c.fill_between(firing_rate, curve_fail, 0, where=(curve_fail < 0), color='lightcoral', alpha=0.15)
ax_c.fill_between(firing_rate, curve_ctrl, 0, where=(curve_ctrl > 0), color='lightskyblue', alpha=0.1)

ax_c.set_title('C. Hippocampal Synaptic Plasticity Curve (BCM modification)', fontsize=11, fontweight='bold')
ax_c.set_xlabel('Post-synaptic Firing Rate (Hz)', fontsize=9)
ax_c.set_ylabel('Synaptic Weight Change ($\Delta W$)', fontsize=9)
ax_c.legend(fontsize=8)
ax_c.grid(True, linestyle=':', alpha=0.4)

fig.suptitle('Figure 2: Downstream Hippocampal Microcircuit Disruption under Hypothalamic Failure', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('Figure_2.png', dpi=300)
plt.show()
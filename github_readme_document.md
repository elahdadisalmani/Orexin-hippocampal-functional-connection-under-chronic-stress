# Orexin Dynamics and Hippocampal Dysregulation in Chronic Stress

A multi-scale computational framework integrating lateral hypothalamic (LH) orexinergic population dynamics with an active hippocampal CA1 spiking neural network (SNN) under chronic stress protocols.

---

## 📌 Overview

Sustained environmental and neuroendocrine stress triggers a cascade of neurobiological alterations that impair hippocampal rhythmicity and cognitive function. This project bridges upstream subcortical dynamics with downstream hippocampal microcircuit computations through a coupled, biophysically grounded architecture:

1. **Upstream Subcortical Module (`upstream_orexin_model.py`):** Captures the temporal biphasic trajectory of lateral hypothalamic (LH) orexinergic neurons under chronic stress using ordinary differential equations (ODEs). It models neuronal population decay ($P_{\text{active}}$), transient homeostatic firing compensation ($f_O$), and cellular resource depletion ($R$).
2. **Downstream Microcircuit Module (`downstream_ca1_snn.py`):** Simulates a 1,000-neuron CA1 spiking neural network (800 regular-spiking excitatory principal cells and 200 fast-spiking PV+ inhibitory interneurons, maintaining a strict 4:1 E/I balance) using computationally efficient Izhikevich dynamics.
3. **Metaplasticity Gating:** Models Bienenstock–Cooper–Munro (BCM) synaptic plasticity with a dynamic sliding modification threshold ($\theta_M$) regulated by instantaneous orexinergic tone ($I_O$).

---

## 📂 Repository Structure

```text
.
├── upstream_orexin_model.py   # ODE integration for LH orexin population dynamics & BCM threshold
├── downstream_ca1_snn.py      # Izhikevich spiking neural network (CA1 microcircuit, LFP & Raster plots)
├── README.md                  # Comprehensive setup and biological overview
```

---

## 🔬 Model Description & Equations

### 1. Upstream Orexin Population Dynamics
* **Active Population Decay:**
  $$\frac{dP_{\text{active}}(t)}{dt} = -\alpha \cdot S(t) \cdot P_{\text{active}}(t)$$
* **Firing Frequency & Resource Consumption:**
  $$\frac{df_O(t)}{dt} = \beta \cdot S(t) \cdot R(t) - \gamma \cdot (f_O(t) - f_{\text{baseline}})$$
  $$\frac{dR(t)}{dt} = \frac{1 - R(t)}{\tau_{\text{rec}}} - \kappa \cdot S(t) \cdot (f_O(t) - f_{\text{baseline}}) \cdot R(t)$$
* **Net Orexinergic Output Drive:**
  $$I_O(t) = P_{\text{active}}(t) \cdot f_O(t) \cdot R(t)$$

### 2. Downstream Hippocampal Microcircuit (CA1 SNN)
* **Membrane Dynamics (Izhikevich Model):**
  $$\frac{dv_i}{dt} = 0.04 v_i^2 + 5 v_i + 140 - u_i + I_{\text{syn}, i} + I_{\text{ext}, i}$$
  $$\frac{du_i}{dt} = a \cdot (b \cdot v_i - u_i)$$
  $$\text{if } v_i \ge 30\text{ mV}, \quad \text{then } v_i \leftarrow c, \; u_i \leftarrow u_i + d$$

---

## ⚙️ Key Model Parameters

All parameter choices are physiologically calibrated based on baseline rodent electrophysiology and computational benchmarks:

| Parameter | Description | Value | Unit |
| :--- | :--- | :--- | :--- |
| $\alpha$ | Orexin population decay rate | $0.04$ | $\text{day}^{-1}$ |
| $\beta$ | Compensatory activation scaling factor | $0.5$ | $\text{Hz} \cdot \text{day}^{-1}$ |
| $\gamma$ | Firing frequency accommodation rate | $0.3$ | $\text{day}^{-1}$ |
| $\tau_{\text{rec}}$ | Metabolic resource recovery time constant | $15.0$ | $\text{days}$ |
| $\kappa$ | Hyperactive metabolic exhaustion rate | $0.08$ | $\text{Hz}^{-1} \cdot \text{day}^{-1}$ |
| $N_e / N_i$ | Excitatory / Inhibitory neuron counts | $800 / 200$ | cells (4:1 E/I ratio) |
| $\sigma_{\text{noise}}$ | Gaussian background noise standard deviation | $0.15$ | -- |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required numerical and visualization packages:

```bash
pip install numpy matplotlib
```

### Running the Simulations

1. **Run the Upstream Orexinergic Trajectory Simulation:**
   ```bash
   python upstream_orexin_model.py
   ```
   *Generates plots illustrating $P_{\text{active}}$, $R(t)$, $f_O(t)$, net drive $I_O(t)$, and the sliding BCM threshold $\theta_M$.*

2. **Run the Downstream CA1 Network (SNN) Simulation:**
   ```bash
   python downstream_ca1_snn.py
   ```
   *Generates spiking raster plots for pyramidal cells and PV+ interneurons alongside local field potential (LFP) approximations.*

---

## 📜 Citation & Reference

If you use this codebase or model architecture in your research, please cite:

```bibtex
@article{ElahdadiSalmani2026,
  title={Orexin neuron population and activity in chronic stress: Computational insights into hippocampal dysregulation},
  author={Elahdadi Salmani, Mahmoud and Nahtani, Mojde},
  journal={Journal of Computational Neuroscience},
  year={2026}
}
```

---

## 📄 License

This repository is licensed under the MIT License - see the `LICENSE` file for details.
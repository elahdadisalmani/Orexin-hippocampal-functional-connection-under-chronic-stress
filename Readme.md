# Orexin Dynamics and Hippocampal Dysregulation in Chronic Stress

This repository contains the computational neuroscience code and simulation scripts for investigating the Orexin-Hippocampal axis and stress-induced CA1 hippocampal microcircuit disruption.

## Author & Attribution
* **Author:** Mahmoud Elahdadi Salmani
* **Domain:** Systems Neuroscience (Orexin-Hippocampal Axis & Synaptic Plasticity)

---

## Repository Structure

* **`Figure 1_3.py`** (Figure 1 Script): 
  Simulates the upstream hypothalamic orexin dynamics, metabolic and vesicular resource depletion, and net synaptic orexin output ($I_O$) over a 120-day chronic stress paradigm (Generates **Figure 1** dual-axis plot).

* **`Figure 2_2.py`** (Figure 2 Script): 
  Simulates the downstream CA1 hippocampal spiking microcircuit disruption, including raster plots for control and late-failure phases, LFP theta oscillations power spectrum (4–8 Hz band), and BCM synaptic plasticity modification curves (Generates **Figure 2** multi-panel analysis).

---

## Prerequisites & Dependencies

Make sure you have Python 3.8+ installed along with the standard scientific computing libraries:

```bash
pip install numpy matplotlib
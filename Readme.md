# Orexin Dynamics and Hippocampal Dysregulation in Chronic Stress

This repository contains the computational neuroscience code and simulation scripts for investigating the Orexin-Hippocampal axis and stress-induced CA1 hippocampal microcircuit disruption.

## Authors & Attribution

* **Mahmoud Elahdadi Salmani**

  * School of Biology, Damghan University, Damghan, Iran

  * Specialty: Neurophysiology

* **Mojde Nahtani**

  * Department of Electrical and Computer Engineering, Semnan University, Semnan, Iran

  * Specialty: Control Engineering

## Repository Structure

* **`Figure 1_3.py`** (Figure 1 Script):
  Simulates the upstream hypothalamic orexin dynamics, metabolic and vesicular resource depletion, and net synaptic orexin output ($I_O$) over a 120-day chronic stress paradigm (Generates **Figure 1** dual-axis plot).

* **`Figure 2_3.py`** (Figure 2 Script):
  Simulates the downstream CA1 hippocampal spiking microcircuit disruption, including raster plots for control and late-failure phases, LFP theta oscillations power spectrum (4–8 Hz band), and BCM synaptic plasticity modification curves (Generates **Figure 2** multi-panel analysis).

## Prerequisites & Dependencies

Make sure you have Python 3.8+ installed along with the required scientific computing libraries:

```
pip install numpy matplotlib

```

## How to Run the Scripts

Both scripts execute non-interactively (`Agg` backend) to process calculations quickly and save publication-ready figures (`.png` at 300 DPI) directly into your working directory.

1. **Run the Upstream Orexin Trajectory Model:**

   ```
   python "Figure 1_3.py"
   
   ```

   *Output:* Generates `Figure_1.png`

2. **Run the Downstream CA1 Microcircuit Model:**

   ```
   python "Figure 2_3.py"
   
   ```

   *Output:* Generates `Figure_2.png`

## License

This project is part of a research manuscript. All rights reserved by the authors.
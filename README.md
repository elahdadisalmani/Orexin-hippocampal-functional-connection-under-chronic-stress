# Orexin neurons, chronic stress and the hippocampal CA1 theta rhythm: model code

Code, parameters and raw results for the article *Orexin neuron population and activity in chronic stress:
Computational insights into hippocampal dysregulation*. The model couples (i) differential equations for the
orexinergic population of the lateral hypothalamus under stress (loss of neurons, compensatory firing, resource
depletion) with (ii) a spiking network of the hippocampal CA1 microcircuit (800 pyramidal cells, 200 interneurons),
and evaluates a BCM plasticity threshold from the simulated rates.

**Every figure and every number quoted in the article is produced by the code in this repository.**
The results are qualitative: parameters are fitted, not measured (see the article, Table 2 and Discussion).

## Layout

| Path | Content |
|---|---|
| `src/orexin_model.py` | upstream model, Eqs. 1-3 and 7 (integration, helper functions) |
| `src/ca1_network.py` | CA1 spiking network, Eqs. 5, 8-11 (RK4, dt = 0.1 ms, numba), LFP and spectra, Eqs. 16-17 |
| `config/parameters.json` | upstream parameters (time unit: day), including the atrophy rate `sigma_A` |
| `config/network_parameters.json` | network parameters (time unit: ms); selection procedure in `_comment` |
| `config/network_parameters_v1.json` | first network, superseded (kept for the supplementary comparison) |
| `scripts/` | one script per analysis or figure (table below) |
| `outputs/` | figures (`.tif`, `.png`), raw results (`.json`, `.csv`, `.npz`) and `key_numbers.json` |
| `expected/key_numbers.json` | reference values of every number quoted in the article |
| `supplement_network_v1/` | results of the superseded first network |
| `run_all.sh` | runs everything in order |

## Requirements

Python 3.12 with `numpy`, `scipy`, `matplotlib`, `pillow` and `numba`
(`pip install -r requirements.txt`; the exact versions used are in `requirements-lock.txt`).
One CPU core is enough.

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_figure1.py          # Figure 1 and upstream results, under 1 minute
```

## Reproducing everything

```bash
bash run_all.sh                         # about 40 minutes on one core
python scripts/check_reproduction.py    # compares outputs/key_numbers.json with expected/key_numbers.json
```

All random numbers are seeded (network seeds 1-5, search seed 2026), so a re-run gives the same values on the same
library versions; different CPUs or library versions can change the last digits. `check_reproduction.py` uses a
relative tolerance of 1e-6 by default (pass a larger value as the first argument if needed).

Optional: `python scripts/search_network.py` re-derives the network selection (344 candidates, about 9 minutes).

### Running in pieces

The long steps can be run in parts, and results are merged into the same output files:

```bash
PARTS=epochs python scripts/run_ca1_analyses.py        # also: sweep | hyst | map (comma separated)
ONLY=full,orexin_only python scripts/run_ablation.py   # also: atrophy_only, atrophy_only_all_synapses, full_all_synapses, recovery
ETAS=0.25,0.375,0.5 python scripts/eta_map.py          # remaining gains: 0.75,1.0,1.5
RESUME=1 python scripts/search_network.py 344          # continues a stopped search
```

Verification: a complete run in a fresh directory (done in pieces as above) reproduced every output of the
original run, with identical JSON and CSV files, identical arrays and pixel-identical figures.

## Figures and tables

| Item in the article | Scripts (in this order) | Main outputs |
|---|---|---|
| Figure 1 (upstream trajectory, withdrawal) | `make_figure1.py` | `figure1_summary.json`, `figure1_timeseries.csv`, `figure1_recovery.csv` |
| Figure 2 (rasters, spectra, BCM) | `run_ca1_analyses.py`, `make_figure2.py` | `ca1_epochs.json`, `ca1_epochs_raw.npz`, `bcm_summary.json` |
| Figure 3 (ablation, withdrawal at network level) | `run_ablation.py`, `make_figure3.py` | `ablation.json`, `ablation_summary.json` |
| Figure 4 (transition, hysteresis, (I_O, A) map) | `run_ca1_analyses.py`, `make_figure4.py` | `ca1_sweep.json`, `ca1_hysteresis.csv`, `ca1_map.npz`, `ca1_transition_summary.json` |
| Supplementary Figure S1 (eta_E) | `eta_map.py`, `make_figureS1.py` | `ca1_eta_map.json` |
| Supplementary Figure S2 (eta_I) | `eta_I_sensitivity.py`, `make_figureS2.py` | `ca1_etaI_sensitivity.json` |
| Supplementary Table S1 (upstream sensitivity) | `upstream_sensitivity.py` | `upstream_sensitivity.json` |
| Supplementary Table S2 (ablation by day) | `run_ablation.py` | `ablation.json` |
| All numbers quoted in the text | `collect_key_numbers.py` | `key_numbers.json` |
| Network selection (Section 2.10) | `search_network.py` | `network_search.json` |

## Equations in the code

| Equations | Where |
|---|---|
| 1-3, 7 (population, firing, resources, output) | `src/orexin_model.py`, function `_rhs` |
| 4, 6 (atrophy A(t), excitatory weights W_0 A) | `A = exp(-sigma_A (day - t_on))` in the analysis scripts; applied by `build_network` in `src/ca1_network.py` (`A_target = "EE"`) |
| 5, 8-10 (synaptic current, Izhikevich neurons, conductances) | `src/ca1_network.py`, function `_run` |
| 11 (orexin-dependent drive) | `src/ca1_network.py`, `I_ext` in `_run` and `run_network` |
| 12-15 (BCM threshold, target rate, modification function) | `scripts/make_figure2.py` (evaluated from simulated rates; weights are static) |
| 16-17 (LFP, theta power) | `analyse` in `src/ca1_network.py` |

## Parameters

`config/parameters.json` and `config/network_parameters.json` hold every parameter; the article's Table 2 lists them
with units and sources. Values that are fitted, assumed or fixed by design are marked as such in the article.
The network weights, baseline drives and inhibitory time constant were selected by `scripts/search_network.py` with
criteria fixed in advance (see the header of the script); of 344 candidates, 4 met the control-state criteria and 1 also
met the orexin criterion.

## Notes

* Synaptic weights are static in the simulations; the BCM rule is evaluated from the simulated firing rates.
* The upstream model runs in days and the network in milliseconds; the network is simulated at selected epochs with
  I_O and A taken from the upstream model.
* `numba` compiles the network on first use (a few seconds).

## Authors and licence

Mahmoud Elahdadi Salmani (School of Biology, Damghan University, Damghan, Iran; neurophysiology) and
Mojde Nahtani (Department of Electrical and Computer Engineering, Semnan University, Semnan, Iran; control engineering).
Code: MIT licence (`LICENSE`). To cite, see `CITATION.cff` and the article; DOI: [to be assigned].

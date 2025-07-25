# FLIP2M Artifact: OASIS Optimization Framework

This repository provides the artifact associated with the paper  
_"FLIP2M: Flexible Intra-layer Parallelism and Inter-layer Pipelining for Multi-Model AR/VR Workloads"_,  
accepted at **CASES 2025**.

➡️ **Please see the [INSTALL.md](INSTALL.md) guide for installation instructions.**

---

## 🧪 Test Suite Overview

This directory contains the full test suite to reproduce the experimental results presented in the paper.  
Executing the test suite generates up to eight CSV files in the `output/` directory, along with the evaluation spreadsheet `eval_flip2m.xlsx`.

The OASIS optimization framework supports three cost objectives:

- **Latency**
- **Energy**
- **Energy–Delay Product (EDP)**

Each experiment below can be reproduced by running the corresponding commands.

---

## 📊 Single-Model Optimization (Figure 11)

Uses the DP-based scheduling engine:

```bash
COST_OBJECTIVE='latency' python -m unittest test_FLIP2M.TestFLIP2M.test_single_model
COST_OBJECTIVE='energy'  python -m unittest test_FLIP2M.TestFLIP2M.test_single_model
COST_OBJECTIVE='EDP'     python -m unittest test_FLIP2M.TestFLIP2M.test_single_model
```
Generates the following CSV files in the `output/` directory:

- `sm_latency_table.csv`
- `sm_energy_table.csv`
- `sm_EDP_table.csv`

---

## 📈 Single-Model Comparison with Tangram (Figure 12)

```bash
python -m unittest test_FLIP2M.TestFLIP2M.test_single_model_TANGRAMcomp
```
Generates:

- `sm_TANGRAMcomp.csv`

---

## 🤖 Multi-Model Optimization (Figure 13)

Uses the CP-SAT constraint solver engine:

```bash
COST_OBJECTIVE='latency' python -m unittest test_FLIP2M.TestFLIP2M.test_multi_model
COST_OBJECTIVE='energy'  python -m unittest test_FLIP2M.TestFLIP2M.test_multi_model
COST_OBJECTIVE='EDP'     python -m unittest test_FLIP2M.TestFLIP2M.test_multi_model
```
Generates:

- `mm_latency_table.csv`
- `mm_energy_table.csv`
- `mm_EDP_table.csv`

---

## 🔁 Multi-Model Comparison with SET (Figure 15)

```bash
python -m unittest test_FLIP2M.TestFLIP2M.test_multi_model_SETcomp
```
Generates:

- `mm_SETcomp.csv`

---

## 📉 Regenerating Paper Figures

To reproduce the figures in the paper:

1. Run the commands above to generate the relevant `.csv` files in the `output/` directory.
2. Open `eval_flip2m.xlsx`.
3. For each red-highlighted cell, paste the corresponding value from the generated CSV files.
4. The embedded charts will automatically update, reproducing Figures 11, 12, 13, and 15 from the paper.

---

## 📎 Citation

If you use this artifact or framework in your work, please cite:

```bibtex
@inproceedings{tombesi2025flip2m,
  title     = {FLIP2M: Flexible Intra-layer Parallelism and Inter-layer Pipelining for Multi-Model AR/VR Workloads},
  author    = {Gabriele Tombesi, Je Yang, Joseph Zuckerman, Davide Giri, William Baisi and Luca P. Carloni},
  year      = {2025}
}
```


# Installation Guide

This document describes the system requirements and setup instructions for running the FLIP2M optimization framework artifact.

---

## ✅ System Requirements

- **Operating System:**  
  - Ubuntu 20.04 / 22.04 / 24.02  
  - CentOS 7

- **Python Version:**  
  - Python 3.8

---

## 🚀 Quick Start

Install Python 3 and `pip` (if not already installed):

```bash
sudo apt install python3
sudo apt install python3-pip
```

## 📦 Install Required Python Packages

Run the following command to install all necessary dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

To test the installation, run the following commands:

```bash
cd FLIP2M
COST_OBJECTIVE="latency" python -m unittest test_FLIP2M.TestFLIP2M.test_single_model_TANGRAMcomp
```

## ✅ Expected Output

If the installation is successful, you should see output similar to the following:

```python
defaultdict(<class 'dict'>,
  { 'resnet50': {
        '2':  {'latency': 8177.0,  'energy': 20026.0,  'dram_acc': 186586066.0},
        '4':  {'latency': 3681.0,  'energy': 12621.0,  'dram_acc': 63041644.0},
        '8':  {'latency': 2115.0,  'energy': 9691.0,   'dram_acc': 62332216.0},
        '16': {'latency': 1850.0,  'energy': 10494.0,  'dram_acc': 63051012.0}},
    'vgg16': {
        '2':  {'latency': 78252.0, 'energy': 159818.0, 'dram_acc': 199256132.0},
        '4':  {'latency': 35742.0, 'energy': 123313.0, 'dram_acc': 174089470.0},
        '8':  {'latency': 17434.0, 'energy': 91002.0,  'dram_acc': 152658384.0},
        '16': {'latency': 9054.0,  'energy': 56905.0,  'dram_acc': 129857928.0}}})
.
----------------------------------------------------------------------
Ran 1 test in 0.126s

OK
```

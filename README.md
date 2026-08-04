# 3D Crossbar-Based Sparse MAC for Hardware-Efficient Neural Network Inference

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue.svg)
![SPICE](https://img.shields.io/badge/SPICE-Circuit%20Simulation-success.svg)
![Research](https://img.shields.io/badge/Project-MSc%20Research-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</p>

---

## Overview

This repository contains the implementation of my **Master of Science (M.Sc.) Final Year Research Project**, focused on the design and evaluation of a **Three-Dimensional Crossbar-Based Sparse Multiply-Accumulate (MAC) Architecture** for energy-efficient neural network inference.

The project investigates how Artificial Neural Networks (ANNs) can be translated from software models into hardware-aware compute-in-memory (CIM) architectures by combining:

- Sparse Neural Networks
- Weight Quantization
- Resistance Mapping
- Crossbar Computing
- Circuit-Level SPICE Simulation

The primary objective is to reduce computational complexity, memory accesses, and hardware resource utilization while maintaining acceptable inference accuracy.

---

# Motivation

Modern Deep Neural Networks require significant computational resources and memory bandwidth.

Traditional Von Neumann architectures suffer from the **Memory Wall**, where the cost of transferring data between processor and memory dominates computation.

Compute-in-Memory (CIM) architectures address this issue by performing matrix-vector multiplication directly inside memory arrays.

This work explores a hardware-aware sparse MAC architecture using a **three-dimensional resistive crossbar model** capable of representing neural network weights as programmable resistance values.

---

# Objectives

The project aims to:

- Design a sparse ANN suitable for hardware implementation
- Reduce memory usage through weight pruning
- Quantize floating-point weights into discrete values
- Convert quantized weights into resistance values
- Generate crossbar-compatible hardware mappings
- Validate the architecture through SPICE simulations
- Compare software and hardware inference performance

---

# Research Workflow

```
Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
ANN Training
      │
      ▼
Weight Pruning
      │
      ▼
Weight Quantization
      │
      ▼
Resistance Mapping
      │
      ▼
Crossbar Netlist Generation
      │
      ▼
SPICE Simulation
      │
      ▼
Hardware Output Validation
```

---

# Repository Structure

```
3d-crossbar-sparse-mac

├── docs/
│   └── MSc Thesis
│
├── hardware/
│   ├── opamp/
│   ├── resistance_tables/
│   ├── spice/
│   └── crossbar_model/
│
├── images/
│
├── python/
│   ├── training/
│   ├── quantization/
│   ├── hardware_mapping/
│   ├── validation/
│   └── utils/
│
├── results/
│   ├── sample_outputs/
│   ├── software/
│   └── spice/
│
└── dataset/
```

---

# Software Pipeline

The software implementation consists of several sequential stages.

## 1. Dataset Preparation

- Dataset loading
- Feature normalization
- Label preprocessing
- Train/Test split

---

## 2. Neural Network Training

Implemented using TensorFlow/Keras.

Features include:

- Dense Feedforward ANN
- ReLU Activation
- Adam Optimizer
- Cross Entropy Loss
- Performance Evaluation

---

## 3. Network Pruning

Low-importance weights are removed to create sparse weight matrices.

Benefits include:

- Reduced memory usage
- Lower hardware complexity
- Reduced switching activity
- Improved energy efficiency

---

## 4. Weight Quantization

Floating-point weights are converted into discrete levels suitable for hardware implementation.

Advantages:

- Reduced storage
- Lower hardware complexity
- Simplified resistance mapping

---

## 5. Resistance Mapping

Quantized weights are translated into equivalent resistance values for the crossbar array.

Separate positive and negative resistance matrices are generated for differential computation.

---

## 6. Crossbar Modeling

The mapped resistance values are used to generate crossbar-compatible structures suitable for SPICE simulation.

---

## 7. Hardware Validation

Generated outputs are compared with software inference results to verify computational correctness.

---

# Hardware Components

## Operational Amplifier

The project includes SPICE simulations of the analog front-end.

Performance evaluated:

- Differential Gain
- Common Mode Gain
- Phase Margin
- Offset Voltage
- Slew Rate
- Transient Response

Simulation plots are available under:

```
hardware/opamp/
```

---

## Resistance Mapping

Discrete resistance tables are generated for each network layer.

Files located under:

```
hardware/resistance_tables/
```

---

## SPICE Simulation

The mapped network is evaluated using SPICE-compatible circuit models.

Simulation validates:

- Current accumulation
- MAC functionality
- Analog computation

---

# Experimental Results

The project demonstrates:

- Successful ANN training
- Sparse network generation
- Weight quantization
- Resistance mapping
- Crossbar generation
- SPICE-level verification
- Hardware output validation

Representative figures are available inside the **images/** directory.

---

# Technologies Used

Programming

- Python
- TensorFlow
- NumPy
- Pandas
- Matplotlib

Hardware

- SPICE
- Analog Crossbar Modeling

Machine Learning

- Artificial Neural Networks
- Weight Pruning
- Quantization

---

# Project Scope

This work focuses on the **software-to-hardware translation** of neural network models into a three-dimensional crossbar architecture.

The implementation concludes with **circuit-level SPICE simulation and validation**.

Physical IC fabrication, RTL implementation, layout generation, and silicon characterization were **outside the scope** of this research.

---

# Limitations

Current limitations include:

- Simulation-only validation
- No fabricated silicon
- No FPGA implementation
- No ASIC synthesis
- Device variability not modeled
- Thermal effects not considered

---

# Future Work

Potential extensions include:

- RTL implementation of MAC architecture
- FPGA acceleration
- OpenLane ASIC implementation
- SkyWater130 tape-out
- Memristor device modeling
- Analog CIM accelerator
- On-chip training
- Hardware-aware quantization

---

# Thesis

The complete MSc thesis is available under:

```
docs/MSc_Thesis_3D_Crossbar_Sparse_MAC.pdf
```

---

# Author

**Goutham Krishnan P.**

M.Sc. Electronics (VLSI Design)

Research Interests

- VLSI Design
- Compute-in-Memory
- AI Hardware
- Analog IC Design
- Mixed Signal Systems
- Neuromorphic Computing

---

# Citation

If you find this work useful, please cite:

```text
Goutham Krishnan P.

"Three-Dimensional Crossbar-Based Sparse MAC for
Hardware-Efficient Neural Network Inference"

M.Sc. Thesis
Digital University Kerala
2025
```

---

# License

This project is released under the MIT License.

---

# Acknowledgements

I would like to thank my project supervisor, faculty members, and the Department of Electronics for their guidance and support throughout this research work.

---

## Repository Status

> ✅ Project Completed

Current Status:

- ✔ Software Pipeline Completed
- ✔ ANN Training
- ✔ Sparse Weight Generation
- ✔ Quantization
- ✔ Resistance Mapping
- ✔ Crossbar Modeling
- ✔ SPICE Simulation
- ✔ Hardware Validation

This repository represents the final research implementation submitted as part of my M.Sc. dissertation.

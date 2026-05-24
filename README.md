# mpyCUDA
Unifying MPI, C++, CUDA, Apache Spark, and Python for maximum parallel processing power! 

| 1st Project: Image Convolution Engine | 2nd Project: AWACS Simulation |
| :---: | :---: |
| [![Open First Project](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Schryzon/mpyCUDA/blob/master/Kamen-Rider-Image-Convolution/colab_notebook.ipynb) | [![Open Second Project](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Schryzon/mpyCUDA/blob/master/Ace-Combat-AWACS-Simulation/colab_notebook.ipynb) |

---

> [!CAUTION]
> ### 🛑 ENABLE YOUR GPU!
> **These projects REQUIRE a CUDA-capable GPU (NVIDIA T4 or better).**
> If you run this on a standard CPU runtime, the code will fail or run extremely slowly.
> 
> **How to fix in Colab:** 
> 1. Go to **Runtime** → **Change runtime type**
> 2. Select **T4 GPU** (or better)
> 3. Click **Save**

---

## 🚀 Quick Start — Google Colab

> **This is the recommended way for lecturers and reviewers.**

1. Click the **Colab Badge** above for the project you wish to evaluate.
2. In Colab, go to **Runtime → Change runtime type → GPU** (T4 is sufficient).
3. Run all cells from top to bottom (`Runtime → Run all`).

---

## 📁 Projects Overview

This repository contains **two parallel programming projects** for the Parallel Programming A course.

1.  **Image Convolution Engine** (ACTIVE)
    -   High-performance image filtering (Blur, Sobel, Sharpen, Emboss).
    -   Uses MPI + CUDA + OpenMP hybrid parallelism.
    -   Implements **Guided Scheduling** for dynamic load balancing.
2.  **Ace Combat AWACS Simulation** (ACTIVE)
    -   Large-scale target prioritization using Apache Spark MLlib (Random Forest).
    -   Advanced Feature Engineering (PySpark Trigonometry) & ML Noise Injection.
    -   Parallel interception trajectory math using CUDA C++.
    -   Atomic GPU operations for massive concurrent thread tracking.
    -   Interactive 3D visualization (Radar Dome & Interception Paths) via Plotly.

### Project Structure
```text
mpyCUDA/
├── Kamen-Rider-Image-Convolution/
│   ├── colab_notebook.ipynb                 ← 1st Project Colab Entry Point
│   ├── images/                              ← Input/output image assets
│   ├── scripts/
│   │   ├── Makefile                         ← Linux build script
│   │   ├── parallel_image.cu                ← The Hybrid MPI+CUDA Engine
│   │   └── parallel_conv.cu                 ← Matrix Convolution Sandbox
│   ├── README.md                            ← Detailed Project Docs
│   └── presentation_guide.md                ← Presentation Outline
│
└── Ace-Combat-AWACS-Simulation/
    ├── colab_notebook.ipynb                 ← Colab Notebook (Linux)
    ├── windows_notebook.ipynb               ← Local Notebook (Windows/Anaconda)
    ├── create_notebook.py                   ← Generator for Colab version
    ├── create_windows_notebook.py           ← Generator for Windows version
    ├── scripts/
    │   ├── build_cuda.bat                   ← Windows MSVC compile script
    │   ├── data_gen.py                      ← 1 Million Record Radar Synthesizer
    │   └── trajectory_math.cu               ← Parallel SAM Trajectory Engine
    └── presentation_guide.md                ← AWACS Presentation Outline
```

## 🛠️ Local Build Instructions

### Project 1: Image Convolution (Linux / WSL)
```bash
cd Kamen-Rider-Image-Convolution/scripts
make all

# Run the primary Image Engine (sobel mode)
mpirun --oversubscribe -n 4 ./parallel_image ../images/input.jpg ../images/output.jpg sobel
```

### Project 2: AWACS Simulation (Windows)
```powershell
cd Ace-Combat-AWACS-Simulation
# Compile the CUDA engine into a dynamic link library (DLL)
./scripts/build_cuda.bat

# Launch the interactive notebook
jupyter notebook windows_notebook.ipynb
```

## ⚡ Core Features

### Hybrid Parallel Architecture (Project 1)
-   **Guided Scheduling**: Dynamic load balancing via MPI Master-Worker pattern to prevent idle cores.
-   **Hybrid Parallelism**: CUDA GPU kernels + OpenMP CPU threads + MPI Clusters working in tandem.
-   **Color Image Support**: Full BGR channel-wise processing.

### Big Data & GPU Interop (Project 2)
-   **Spark MLlib**: Distributed processing of 1,000,000 radar records to predict critical threats.
-   **PySpark Feature Engineering**: Vectorized trigonometry (`atan2`) for accurate model training.
-   **CUDA Atomics**: Thread-safe global evasion counters natively processed on the GPU.
-   **Ctypes Integration**: Seamless passing of data arrays from Python directly to CUDA VRAM.

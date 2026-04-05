# 🏍️ mpyCUDA: Henshin!!!
Unifying MPI, C++, CUDA, and Jupyter for maximum parallel processing power! 

> *"Obaachan ga itte ita... Ten no michi o yuki, subete o tsukasadoru otoko."* (Grandmother used to say... by walking through the path of heaven, you will be the man who will rule everything.) — Kamen Rider Kabuto

| 1st Project: Kamen Rider Image Conv | 2nd Project: [Planned] |
| :---: | :---: |
| [![Open First Project](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Schryzon/mpyCUDA/blob/master/Kamen-Rider-Image-Convolution/colab_notebook.ipynb) | [![Open Second Project](https://colab.research.google.com/assets/colab-badge.svg)](#) |

---

> [!CAUTION]
> ### 🛑 HENSHIN IMPOSSIBLE: ENABLE YOUR GPU!
> **This project REQUIRES a CUDA-capable GPU (NVIDIA T4 or better).**
> If you run this on a standard CPU runtime, the code will fail to process images.
> 
> **How to fix:** 
> 1. Go to **Runtime** → **Change runtime type**
> 2. Select **T4 GPU** (or better)
> 3. Click **Save**
> 
> *Obaachan ga itte ita... "You cannot walk the path of heaven if your engine is not even turned on."*

---

## 🚀 Quick Start — Google Colab

> **This is the recommended way for lecturers and reviewers.**

1. Click the **"Open First Project"** badge above.
2. In Colab, go to **Runtime → Change runtime type → GPU** (T4 is sufficient).
3. Run all cells from top to bottom (`Runtime → Run all`).

---

## 📁 Projects Overview

This repository will contain **two parallel programming projects** for the Parallel Programming A course.

1.  **Kamen Rider Image Convolution** (ACTIVE)
    -   High-performance image filtering (Blur, Sobel, Sharpen, Emboss).
    -   Uses MPI + CUDA + OpenMP hybrid parallelism.
    -   Implements **Guided Scheduling** for dynamic load balancing.
2.  **Second Project** (COMING SOON)
    -   Stay tuned for the next parallel computing evolution!

### Structure
```
mpyCUDA/
└── Kamen-Rider-Image-Convolution/
    ├── colab_notebook.ipynb                 ← 1st Project Colab Entry Point
    ├── images/                              ← Input/output image assets
    ├── scripts/
    │   ├── Makefile                         ← Linux build script
    │   ├── parallel_image.cu                ← The Hybrid MPI+CUDA Engine
    │   └── parallel_conv.cu                 ← Matrix Convolution Sandbox
    ├── README.md                            ← Detailed Project Docs
    └── presentation_guide.md                ← (Hidden) Presentation Outline
```

## 🛠️ Local Build (Linux/WSL)

```bash
cd Kamen-Rider-Image-Convolution/scripts
make all

# Run the primary Image Engine (sobel mode)
mpirun --oversubscribe -n 4 ./parallel_image ../images/input.jpg ../images/output.jpg sobel
```

## Features

-   **Guided Scheduling**: Dynamic load balancing via MPI Master-Worker pattern.
-   **Hybrid Parallelism**: CUDA GPU kernels + OpenMP CPU threads + MPI Clusters.
-   **Color Image Support**: Full BGR channel-wise processing.
-   **Henshin Ready**: Optimized for speed and performance.

# mpyCUDA
Unifying MPI, C++, CUDA, and Jupyter for maximum parallel processing understanding!!!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Schryzon/mpyCUDA/blob/main/colab_run.ipynb)

---

## 🚀 Quick Start — Google Colab

> **This is the recommended way for lecturers and reviewers.**

1. Click the **"Open in Colab"** badge above.
2. In Colab, go to **Runtime → Change runtime type → GPU** (T4 is sufficient).
3. Run all cells from top to bottom (`Runtime → Run all`).

The notebook will automatically:
- Mount your Google Drive and save the repo to `My Drive/Jay-IF24-mpyCUDA`
- Install all system dependencies (OpenMPI, OpenCV)
- Compile the CUDA/MPI programs using the Linux `Makefile`
- Run benchmarks and generate plots

---

## 📁 Project Structure

```
mpyCUDA/
├── colab_run.ipynb                          ← Main Colab entry point (start here!)
└── Kamen-Rider-Image-Convolution/
    ├── images/                              ← Input/output images
    ├── scripts/
    │   ├── Makefile                         ← Linux build script (for Colab)
    │   ├── parallel_image.cu                ← MPI+CUDA image processing (blur/edge)
    │   └── parallel_conv.cu                 ← MPI+CUDA matrix convolution
    ├── notebook_image_linux.ipynb           ← Standalone image processing notebook
    └── notebook_linux.ipynb                 ← Standalone convolution notebook
```

## 🛠️ Local Build (Linux/WSL)

```bash
cd Kamen-Rider-Image-Convolution/scripts
make all

# Run image processing (blur mode, 4 ranks)
mpirun --oversubscribe -n 4 ./parallel_image ../images/input.jpg ../images/output_blur.jpg blur

# Run matrix convolution (4 ranks, 2048x2048)
mpirun --oversubscribe -n 4 ./parallel_conv 2048
```

## Features

- **Guided Scheduling**: Dynamic load balancing via MPI Master-Worker pattern
- **Multi-mode**: Box Blur and Sobel Edge Detection
- **Hybrid parallelism**: CUDA GPU kernels + OpenMP CPU threads
- **Color image support**: Full BGR channel-wise processing

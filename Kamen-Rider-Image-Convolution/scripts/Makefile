# =============================================================================
# Makefile for mpyCUDA - Linux / Google Colab
# Mirrors the build settings from .vscode/tasks.json (translated to Linux/GCC)
# =============================================================================

NVCC       := nvcc
STD        := c++17
OPT_FLAGS  := --use_fast_math -lineinfo -std=$(STD) -O2 -arch=sm_75

# Warning Suppression
WARN_FLAGS := -w -Xcompiler -w

# OpenMP: translate -Xcompiler /openmp (MSVC) → -Xcompiler -fopenmp (GCC)
OMP_FLAGS  := -Xcompiler -fopenmp

# MPI: use system openmpi (replaces MS-MPI)
MPI_INC    := $(shell mpicxx --showme:compile)
MPI_LIBS   := $(shell mpicxx --showme:link)

# OpenCV: use pkg-config (replaces hardcoded OpenCV paths)
OPENCV_INC := $(shell pkg-config --cflags opencv4)
OPENCV_LIB := $(shell pkg-config --libs opencv4)

# Combine all flags
INCLUDES   := $(MPI_INC) $(OPENCV_INC)
LIBS       := $(MPI_LIBS) $(OPENCV_LIB) -lgomp

# Targets
all: parallel_image parallel_conv

parallel_image: parallel_image.cu
	$(NVCC) $(OPT_FLAGS) $(OMP_FLAGS) $(WARN_FLAGS) $(INCLUDES) $< -o $@ $(LIBS)

parallel_conv: parallel_conv.cu
	$(NVCC) $(OPT_FLAGS) $(OMP_FLAGS) $(WARN_FLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -f parallel_image parallel_conv

.PHONY: all clean

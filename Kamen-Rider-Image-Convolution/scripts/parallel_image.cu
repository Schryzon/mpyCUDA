#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define COMM MPI_COMM_WORLD
#define TAG_RANGE 1
#define TAG_DATA 2
#define TAG_RESULT 3
#define TAG_DIE 4
#define TAG_READY 5

// Box Blur Kernel
__global__ void blur_kernel(unsigned char *input, unsigned char *output,
                            int width, int height, int start_row, int chunk_rows, int offset, int channels, float strength) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < chunk_rows) {
    int global_y = start_row + y;
    int local_idx = (y + offset) * width * channels + x * channels;
    
    // Check for global image boundaries
    if (global_y == 0 || global_y == height - 1 || x == 0 || x == width - 1) {
      for (int c = 0; c < channels; ++c) {
        output[local_idx + c] = input[local_idx + c];
      }
      return;
    }

    float alpha = strength / 10.0f; // Normalize strength 1-10 to 0.1-1.0
    for (int c = 0; c < channels; ++c) {
      int cur_idx = local_idx + c;
      int sum = input[cur_idx] + 
                input[cur_idx - channels] + 
                input[cur_idx + channels] +
                input[cur_idx - width * channels] + 
                input[cur_idx + width * channels] +
                input[cur_idx - width * channels - channels] + 
                input[cur_idx - width * channels + channels] +
                input[cur_idx + width * channels - channels] + 
                input[cur_idx + width * channels + channels];
      float blurred = (float)sum / 9.0f;
      float original = (float)input[cur_idx];
      output[cur_idx] = (unsigned char)((1.0f - alpha) * original + alpha * blurred);
    }
  }
}

// basic Edge Detection Kernel
__global__ void edge_kernel(unsigned char *input, unsigned char *output,
                            int width, int height, int start_row, int chunk_rows, int offset, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < chunk_rows) {
    int global_y = start_row + y;
    int local_idx = (y + offset) * width * channels + x * channels;

    // Check for global image boundaries
    if (global_y == 0 || global_y == height - 1 || x == 0 || x == width - 1) {
      for (int c = 0; c < channels; ++c) {
        output[local_idx + c] = 0; // Standard for edge detection
      }
      return;
    }

    for (int c = 0; c < channels; ++c) {
      int cur_idx = local_idx + c;
      int dx = input[cur_idx + channels] - input[cur_idx - channels];
      int dy = input[cur_idx + width * channels] - input[cur_idx - width * channels];
      float val = sqrtf((float)(dx * dx + dy * dy));
      output[cur_idx] = (val > 255.0f) ? 255 : (unsigned char)val;
    }
  }
}

// Sobel Edge Detection Kernel
__global__ void sobel_kernel(unsigned char *input, unsigned char *output,
                             int width, int height, int start_row, int chunk_rows, int offset, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < chunk_rows) {
    int global_y = start_row + y;
    int local_idx = (y + offset) * width * channels + x * channels;

    // Check for global image boundaries
    if (global_y == 0 || global_y == height - 1 || x == 0 || x == width - 1) {
      for (int c = 0; c < channels; ++c) {
        output[local_idx + c] = 0;
      }
      return;
    }

    for (int c = 0; c < channels; ++c) {
      int cur_idx = local_idx + c;
      
      float Gx = -1.0f * (float)input[cur_idx - width * channels - channels] + 1.0f * (float)input[cur_idx - width * channels + channels] +
                 -2.0f * (float)input[cur_idx - channels] + 2.0f * (float)input[cur_idx + channels] +
                 -1.0f * (float)input[cur_idx + width * channels - channels] + 1.0f * (float)input[cur_idx + width * channels + channels];
      
      float Gy = -1.0f * (float)input[cur_idx - width * channels - channels] - 2.0f * (float)input[cur_idx - width * channels] - 1.0f * (float)input[cur_idx - width * channels + channels] +
                  1.0f * (float)input[cur_idx + width * channels - channels] + 2.0f * (float)input[cur_idx + width * channels] + 1.0f * (float)input[cur_idx + width * channels + channels];
      
      float val = sqrtf(Gx * Gx + Gy * Gy);
      output[cur_idx] = (val > 255.0f) ? 255 : (unsigned char)val;
    }
  }
}

// Sharpen Kernel
__global__ void sharpen_kernel(unsigned char *input, unsigned char *output,
                               int width, int height, int start_row, int chunk_rows, int offset, int channels, float strength) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < chunk_rows) {
    int global_y = start_row + y;
    int local_idx = (y + offset) * width * channels + x * channels;

    // Check for global image boundaries
    if (global_y == 0 || global_y == height - 1 || x == 0 || x == width - 1) {
      for (int c = 0; c < channels; ++c) {
        output[local_idx + c] = input[local_idx + c];
      }
      return;
    }

    float alpha = strength / 5.0f; // Standard strength (5) is 1.0 multiplier
    for (int c = 0; c < channels; ++c) {
      int cur_idx = local_idx + c;
      
      float original = (float)input[cur_idx];
      float neighbors_sum = (float)input[cur_idx - channels] + 
                            (float)input[cur_idx + channels] +
                            (float)input[cur_idx - width * channels] + 
                            (float)input[cur_idx + width * channels];
      
      // Standard sharpen: 5 * original - neighbors = original + (4 * original - neighbors)
      float detail = 4.0f * original - neighbors_sum;
      float final_val = original + alpha * detail;
      
      output[cur_idx] = (final_val > 255.0f) ? 255 : (final_val < 0.0f) ? 0 : (unsigned char)final_val;
    }
  }
}

// Emboss Kernel
__global__ void emboss_kernel(unsigned char *input, unsigned char *output,
                               int width, int height, int start_row, int chunk_rows, int offset, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < chunk_rows) {
    int global_y = start_row + y;
    int local_idx = (y + offset) * width * channels + x * channels;

    // Check for global image boundaries
    if (global_y == 0 || global_y == height - 1 || x == 0 || x == width - 1) {
      for (int c = 0; c < channels; ++c) {
        output[local_idx + c] = input[local_idx + c];
      }
      return;
    }

    for (int c = 0; c < channels; ++c) {
      int cur_idx = local_idx + c;
      
      float sum = 1.0f * (float)input[cur_idx] +
                 -2.0f * (float)input[cur_idx - width * channels - channels] - 1.0f * (float)input[cur_idx - width * channels] +
                 -1.0f * (float)input[cur_idx - channels] + 1.0f * (float)input[cur_idx + channels] +
                  1.0f * (float)input[cur_idx + width * channels] + 2.0f * (float)input[cur_idx + width * channels + channels];
      
      sum += 128.0f;
      output[cur_idx] = (sum > 255.0f) ? 255 : (sum < 0.0f) ? 0 : (unsigned char)sum;
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  printf("Rank %d started\n", rank);

  if (argc < 4) {
    if (rank == 0) {
      std::cerr
          << "Usage: mpirun -n <N> ./parallel_image <input> <output> <mode> [strength (1-10)]"
          << std::endl;
      std::cerr << "Modes: blur, edge, sobel, sharpen, emboss" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  std::string mode = argv[3];
  float strength = 5.0f;
  if (argc >= 5) {
    strength = std::stof(argv[4]);
    if (strength < 1.0f) strength = 1.0f;
    if (strength > 10.0f) strength = 10.0f;
  }

  cv::Mat image;
  int width, height, channels;

  if (rank == 0) {
    image = cv::imread(argv[1]);
    if (image.empty()) {
      std::cerr << "Failed to load image!" << std::endl;
      MPI_Abort(COMM, 1);
    }
    width = image.cols;
    height = image.rows;
    channels = image.channels();
    std::cout << "Image Loaded: " << width << "x" << height << " channels: " << channels << " Mode: " << mode
              << " Strength: " << strength << " with " << size << " ranks." << std::endl;
    std::cout << "Using GUIDED scheduling for load balancing." << std::endl;
  }

  MPI_Bcast(&width, 1, MPI_INT, 0, COMM);
  MPI_Bcast(&height, 1, MPI_INT, 0, COMM);
  MPI_Bcast(&channels, 1, MPI_INT, 0, COMM);
  MPI_Bcast(&strength, 1, MPI_FLOAT, 0, COMM);

  if (rank == 0) {
    // Scheduler (Master)
    int next_row = 0;
    int workers_active = size - 1;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Simple worker loop for master if size=1 (not recommended)
    if (size == 1) {
      // Master does all work if only one rank exists (edge case)
      std::cerr << "Please run with at least 2 ranks for Master-Worker logic."
                << std::endl;
    } else {
      while (workers_active > 0) {
        int range[2];
        MPI_Status status;
        // Wait for a worker to request work (Send result tag or initial
        // request)
        // We receive up to 2 ints to avoid truncation error for TAG_RESULT
        MPI_Recv(range, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, COMM,
                 &status);
        int worker = status.MPI_SOURCE;

        if (status.MPI_TAG == TAG_RESULT) {
          int start_r = range[0];
          int r_count = range[1];
          MPI_Recv(image.ptr(start_r), r_count * width * channels, MPI_UNSIGNED_CHAR,
                   worker, TAG_RESULT, COMM, &status);
        }
        // If TAG_READY or after processing TAG_RESULT, send new work

        if (next_row < height) {
          // Calculate Guided chunk size
          int remaining = height - next_row;
          int chunk_size = std::max(16, remaining / (2 * (size - 1)));
          if (next_row + chunk_size > height)
            chunk_size = height - next_row;

          int range[2] = {next_row, chunk_size};
          MPI_Send(range, 2, MPI_INT, worker, TAG_RANGE, COMM);

          // Correct Halo Strategy:
          // Processing range [next_row, next_row + chunk_size - 1]
          // Top halo: next_row - 1 (if it exists)
          // Bottom halo: next_row + chunk_size (if it exists)
          int send_start = std::max(0, next_row - 1);
          int send_end = std::min(height - 1, next_row + chunk_size);
          int actual_rows = send_end - send_start + 1;
          int offset = next_row - send_start;

          int header[2] = {actual_rows, offset}; // row count, data start offset
          MPI_Send(header, 2, MPI_INT, worker, TAG_DATA, COMM);
          MPI_Send(image.ptr(send_start), actual_rows * width * channels,
                   MPI_UNSIGNED_CHAR, worker, TAG_DATA, COMM);

          next_row += chunk_size;
        } else {
          // Kill worker
          MPI_Send(nullptr, 0, MPI_INT, worker, TAG_DIE, COMM);
          workers_active--;
        }
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration<double>(end_time - start_time).count();
    cv::imwrite(argv[2], image);
    printf("Time taken: %f\n", elapsed);

  } else {
    // Worker
    // Initial request using TAG_READY
    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, COMM);

    // Pre-allocate GPU buffers for reuse
    unsigned char *d_input = nullptr, *d_output = nullptr;
    size_t current_gpu_size = 0;

    while (true) {
      MPI_Status status;
      int range[2];
      MPI_Recv(range, 2, MPI_INT, 0, MPI_ANY_TAG, COMM, &status);

      if (status.MPI_TAG == TAG_DIE)
        break;

      int start_row = range[0];
      int chunk_rows = range[1];

      int header[2];
      MPI_Recv(header, 2, MPI_INT, 0, TAG_DATA, COMM, &status);
      int total_rows = header[0];
      int offset = header[1];

      std::vector<unsigned char> local_data(total_rows * width * channels);
      MPI_Recv(local_data.data(), total_rows * width * channels, MPI_UNSIGNED_CHAR, 0,
               TAG_DATA, COMM, &status);

      printf("Rank %d processing rows %d to %d\n", rank, start_row,
             start_row + chunk_rows - 1);

      // GPU Processing
      size_t needed_size = (size_t)total_rows * width * channels;
      if (needed_size > current_gpu_size) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_input, needed_size);
        cudaMalloc(&d_output, needed_size);
        current_gpu_size = needed_size;
      }

      cudaMemcpy(d_input, local_data.data(), needed_size, cudaMemcpyHostToDevice);

      dim3 threads(16, 16);
      dim3 blocks((width + 15) / 16, (chunk_rows + 15) / 16);

// Use threads for pre-processing demo
#pragma omp parallel for
      for (int i = 0; i < 100; ++i) { /* dummy host work */
      }

      if (mode == "blur") {
        blur_kernel<<<blocks, threads>>>(d_input, d_output, width, height, start_row, chunk_rows,
                                         offset, channels, strength);
      } else if (mode == "edge") {
        edge_kernel<<<blocks, threads>>>(d_input, d_output, width, height, start_row, chunk_rows,
                                         offset, channels);
      } else if (mode == "sobel") {
        sobel_kernel<<<blocks, threads>>>(d_input, d_output, width, height, start_row, chunk_rows,
                                          offset, channels);
      } else if (mode == "sharpen") {
        sharpen_kernel<<<blocks, threads>>>(d_input, d_output, width, height, start_row, chunk_rows,
                                            offset, channels, strength);
      } else if (mode == "emboss") {
        emboss_kernel<<<blocks, threads>>>(d_input, d_output, width, height, start_row, chunk_rows,
                                           offset, channels);
      } else {
        // Default to blur if mode is unknown
        blur_kernel<<<blocks, threads>>>(d_input, d_output, width, height, start_row, chunk_rows,
                                         offset, channels, strength);
      }
      cudaDeviceSynchronize();
      cudaGetLastError();
      cudaMemcpy(local_data.data(), d_output, needed_size,
                 cudaMemcpyDeviceToHost);

      // Send back only the processed part
      MPI_Send(range, 2, MPI_INT, 0, TAG_RESULT, COMM);
      MPI_Send(local_data.data() + (offset * width * channels), chunk_rows * width * channels,
               MPI_UNSIGNED_CHAR, 0, TAG_RESULT, COMM);
    }
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
  }

  printf("Rank %d finished\n", rank);
  MPI_Finalize();
  return 0;
}
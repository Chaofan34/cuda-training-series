#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>

// these are just for timing measurments
#include "include/timer.h"
#include "include/error.h"

using namespace std;
const int DSIZE = 8192;
const int block_size = 32; // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

__device__ inline float index(const float *A, uint y, uint x, uint ds)
{
  return A[y * ds + x];
}

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul_slow(const float *A, const float *B, float *C, int ds)
{

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds))
  {
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[idx * ds + i] * B[i * ds + idy]; // dot product of row and column
    C[idy * ds + idx] = temp;
  }
}

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds)
{

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds))
  {
    float temp = 0;
    for (int i = 0; i < ds / block_size; i++)
    {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = index(A, idy, i * block_size + threadIdx.x, ds); // load idy row, col is change
      Bs[threadIdx.y][threadIdx.x] = index(B, i * block_size + threadIdx.y, idx, ds); // load idx col, row is change

      // Synchronize
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
        temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();
    }

    // Write to global memory
    C[idy * ds + idx] = temp;
  }
}

int main(int argc, char *argv[])
{

  std::vector<std::string> argvs;
  for (int i = 0; i < argc; ++i)
  {
    argvs.push_back(string(argv[i]));
  }
  bool is_perf = find(argvs.begin(), argvs.end(), "perf") != argvs.end();

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  {
    auto timer = TimeMonitor("BeginCompute");
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    for (int i = 0; i < DSIZE * DSIZE; i++)
    {
      h_A[i] = A_val;
      h_B[i] = B_val;
      h_C[i] = 0;
    }
  }

  {
    auto timer = TimeMonitor(string("GPU Compute, ") + string(is_perf ? "perf" : "slow"));
    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Cuda processing sequence step 1 is complete

    // Launch kernel
    dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    if (is_perf)
    {
      mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    }
    else
    {
      mmul_slow<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    }

    cudaCheckErrors("kernel launch failure");

    // Cuda processing sequence step 2 is complete

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Cuda processing sequence step 3 is complete
  }

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE * DSIZE; i++)
    if (h_C[i] != A_val * B_val * DSIZE)
    {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
      return -1;
    }
  printf("Success!\n");
  return 0;
}

// performance
// $ ./matrix_mul_shared perf
// TimeMonitor::BeginCompute, took 0.226819 seconds
// TimeMonitor::GPU Compute, perf, took 1.23077 seconds
// Success!

// $ ./matrix_mul_shared
// TimeMonitor::BeginCompute, took 0.247985 seconds
// TimeMonitor::GPU Compute, slow, took 10.6584 seconds
// Success!
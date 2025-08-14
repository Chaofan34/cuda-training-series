#include <stdio.h>

// these are just for timing measurments
#include <time.h>
#include "include/error.h"
#include "include/timer.h"

const int DSIZE = 4096;
const int block_size = 16; // CUDA maximum is 1024 *total* threads in block
const float A_val = 1.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds)
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

void mmul_host(const float *A, const float *B, float *C, int ds)
{
  for (int i = 0; i < ds; i++)
  {
    // printf("mul_host: i:%d\n", i);
    for (int j = 0; j < ds; j++)
      for (int k = 0; k < ds; k++)
        C[i * ds + j] += A[i * ds + k] * B[k * ds + j];
  }
}

int main()
{

  float *h_A, *h_B, *h_C, *h_D, *d_A, *d_B, *d_C;

  // these are just for timing

  {
    auto timer = TimeMonitor("Begin Compute");
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];
    h_D = new float[DSIZE * DSIZE];
    for (int i = 0; i < DSIZE * DSIZE; i++)
    {
      h_A[i] = A_val;
      h_B[i] = B_val;
      h_C[i] = 0;
    }
  }

  {
    auto timer = TimeMonitor("GPU Compute");
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
    mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // Cuda processing sequence step 2 is complete
    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  }

  // CPU Compute
  {
    auto timer = TimeMonitor("CPU Compute");
    mmul_host(h_A, h_B, h_D, DSIZE);
  }

  // Verify results
  for (int i = 0; i < DSIZE * DSIZE; i++)
  {
    if (h_C[i] != A_val * B_val * DSIZE)
    {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
      return -1;
    }
    if (h_D[i] != h_C[i])
    {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_D[i], h_C[i]);
      return -1;
    }
  }
  printf("Success!\n");
  return 0;
}

// $ ./matrix_mul, 没有开优化，CPU频率3GHZ，不懂为啥需要978s
// TimeMonitor::Begin Compute, took 0.056859 seconds
// TimeMonitor::GPU Compute, took 0.783124 seconds
// TimeMonitor::CPU Compute, took 978.699 seconds

/*
CPU代码慢的分析过程，尝试查看cpu代码的汇编版本
1. nvcc是将cu文件拆成cuda代码和host代码，cuda代码nvcc编译成ptx, host代码通过gcc/g++编译.s, 可以通过下面命令查看汇编版本
# PTX
nvcc -ptx matrix_mul.cu -o matrix_mul.ptx
# Host 汇编
nvcc -c -Xcompiler -S matrix_mul.cu -o matrix_mul_host.s

2. 使用chatgpt分析汇编代码: https://chatgpt.com/s/t_689e080327648191bfb4ee0a47b51657
- 使用函数名搜索.s文件，找到mmul_host函数的汇编代码, 让chatgpt分析代码
- .cfi_startproc / .cfi_endproc: 调试信息（栈回溯用）
- .section	.text. 为代码段
*/
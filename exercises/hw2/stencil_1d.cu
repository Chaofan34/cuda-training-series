#include <stdio.h>
#include <algorithm>
#include <sstream>
#include "include/timer.h"

using namespace std;

#define N 4096 * 4096 * 8
#define RADIUS 64
#define BLOCK_SIZE 128

__global__ void stencil_1d_slow(int *in, int *out)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (0 <= tid < N)
  {
    int gid = tid + RADIUS;
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
    {
      result += in[offset + gid];
    }
    out[gid] = result;
  }
}

__global__ void stencil_1d_perf1(int *in, int *out)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x;

  // Read input elements into shared memory
  // cudaNote: RADIS Must Lt BLOCKSIZE
  temp[lindex + RADIUS] = in[gindex + RADIUS];
  if (threadIdx.x < RADIUS)
  {
    temp[lindex] = in[gindex];
    temp[lindex + BLOCK_SIZE + RADIUS] = in[gindex + BLOCK_SIZE + RADIUS];
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++)
    result += temp[offset + lindex + RADIUS];

  // Store the result
  out[gindex + RADIUS] = result;
}

void fill_ints(int *x, int n)
{
  fill_n(x, n, 1);
}

int main(int argc, char *argv[])
{
  int *in, *out;     // host copies of a, b, c
  int *d_in, *d_out; // device copies of a, b, c

  // Alloc space for host copies and setup values
  int size = (N + 2 * RADIUS) * sizeof(int);
  in = (int *)malloc(size);
  fill_ints(in, N + 2 * RADIUS);
  out = (int *)malloc(size);
  fill_ints(out, N + 2 * RADIUS);

  // Alloc space for device copies
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Copy to device
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  // bool not set value, count may be rand num
  bool use_perf = false;
  for (int i = 0; i < argc; ++i)
  {
    if (string(argv[i]) == string("perf"))
    {
      use_perf = true;
    }
    std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
  }

  std::cout << "use_perf[" << use_perf << "]" << std::endl;
  // Launch stencil_1d() kernel on GPU
  {
    std::stringstream ss;
    ss << "stencil_1d:" << use_perf;

    // cudaNote: 当前模板计算量太小，感觉跑出来区别不大，可能是slow版本的缓存命中率也挺高, 和share_memory差别不大
    auto timer = TimeMonitor(ss.str());
    if (use_perf == true)
    {
      stencil_1d_perf1<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out);
    }
    else
    {
      stencil_1d_slow<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out);
    }
  }

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < N + 2 * RADIUS; i++)
  {
    if (i < RADIUS || i >= N + RADIUS)
    {
      if (out[i] != 1)
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
    }
    else
    {
      if (out[i] != 1 + 2 * RADIUS)
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2 * RADIUS);
    }
  }

  // Cleanup
  free(in);
  free(out);
  cudaFree(d_in);
  cudaFree(d_out);
  printf("Success!\n");
  return 0;
}

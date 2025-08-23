#include <math.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include "include/error.h"
#include "include/timer.h"

#define USE_STREAMS

// modifiable
typedef float ft;
const int chunks = 64;
const size_t ds = 1024 * 1024 * chunks;
const int count = 22;
const int num_streams = 8;

// not modifiable
const float sqrt_2PIf = 2.5066282747946493232942230134974f;
const double sqrt_2PI = 2.5066282747946493232942230134974;
__device__ float gpdf(float val, float sigma)
{
  return expf(-0.5f * val * val) / (sigma * sqrt_2PIf);
}

__device__ double gpdf(double val, double sigma)
{
  return exp(-0.5 * val * val) / (sigma * sqrt_2PI);
}

// compute average gaussian pdf value over a window around each point
__global__ void gaussian_pdf(const ft *__restrict__ x, ft *__restrict__ y, const ft mean, const ft sigma, const int n)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n)
  {
    ft in = x[idx] - (count / 2) * 0.01f;
    ft out = 0;
    for (int i = 0; i < count; i++)
    {
      ft temp = (in - mean) / sigma;
      out += gpdf(temp, sigma);
      in += 0.01f;
    }
    y[idx] = out / count;
  }
}

// host-based timing
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

int main()
{
  ft *h_x, *d_x, *h_y, *h_y1, *d_y;
  cudaHostAlloc(&h_x, ds * sizeof(ft), cudaHostAllocDefault);
  cudaHostAlloc(&h_y, ds * sizeof(ft), cudaHostAllocDefault);
  cudaHostAlloc(&h_y1, ds * sizeof(ft), cudaHostAllocDefault);
  cudaMalloc(&d_x, ds * sizeof(ft));
  cudaMalloc(&d_y, ds * sizeof(ft));
  cudaCheckErrors("allocation error");

  cudaStream_t streams[num_streams];
  for (int i = 0; i < num_streams; i++)
  {
    cudaStreamCreate(&streams[i]);
  }
  cudaCheckErrors("stream creation error");

  gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds); // warm-up

  for (size_t i = 0; i < ds; i++)
  {
    h_x[i] = rand() / (ft)RAND_MAX;
  }
  cudaDeviceSynchronize();

  {
    auto t = TimeMonitor("no_stream");
    cudaMemcpy(d_x, h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);
    gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds);
    cudaMemcpy(h_y1, d_y, ds * sizeof(ft), cudaMemcpyDeviceToHost);
    cudaCheckErrors("non-streams execution error");
  }

#ifdef USE_STREAMS
  cudaMemset(d_y, 0, ds * sizeof(ft));
  {
    auto t = TimeMonitor("stream");
    for (int i = 0; i < chunks; i++)
    { // depth-first launch
      int stream_idx = i % num_streams;
      int chunk_size = ds / chunks;
      int data_idx = i * chunk_size;
      cudaMemcpyAsync(d_x + data_idx, h_x + data_idx, (chunk_size) * sizeof(ft), cudaMemcpyHostToDevice, streams[stream_idx]);
      gaussian_pdf<<<((chunk_size) + 255) / 256, 256, 0, streams[stream_idx]>>>(d_x + data_idx, d_y + data_idx, 0.0, 1.0, ds);
      cudaMemcpyAsync(h_y + data_idx, d_y + data_idx, chunk_size * sizeof(ft), cudaMemcpyDeviceToHost, streams[stream_idx]);
    }
    cudaDeviceSynchronize();
    cudaCheckErrors("streams execution error");
  }
  for (int i = 0; i < ds; i++)
  {
    if (h_y[i] != h_y1[i])
    {
      std::cout << "mismatch at " << i << " was: " << h_y[i] << " should be: " << h_y1[i] << std::endl;
      return -1;
    }
  }

#endif

  return 0;
}

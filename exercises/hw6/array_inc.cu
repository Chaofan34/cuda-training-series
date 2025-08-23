#include <cstdio>
#include <cstdlib>
#include "include/error.h"
#include "include/timer.h"

template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes)
{
  cudaMallocManaged(&ptr, num_bytes);
}

__global__ void inc(int *array, size_t n)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n)
  {
    array[idx]++;
    idx += blockDim.x * gridDim.x; // grid-stride loop
  }
}

const size_t ds = 1024ULL * 1024ULL * 1024ULL;

const int iterate_size = 1;

int main(int argc, char *argv[])
{
  int *h_array;
  auto size = ds * sizeof(int);
  alloc_bytes(h_array, size);
  memset(h_array, 0, size);

  bool prefetch = (argc > 1 && std::string(argv[1]) == "prefetch");
  if (prefetch)
  {
    auto t = TimeMonitor("prefetch");
    cudaMemPrefetchAsync(h_array, size, 0);
    for (auto i = 0; i < iterate_size; i++)
    {
      inc<<<256, 256>>>(h_array, ds);
      cudaCheckErrors("kernel launch error");
    }
    cudaMemPrefetchAsync(h_array, size, cudaCpuDeviceId);
    cudaDeviceSynchronize();
  }
  else
  {
    auto t = TimeMonitor("no_prefetch");
    for (auto i = 0; i < iterate_size; i++)
    {
      inc<<<256, 256>>>(h_array, ds);
    }
    cudaCheckErrors("kernel launch error");
    cudaDeviceSynchronize();
  }

  for (int i = 0; i < ds; i++)
    if (h_array[i] != iterate_size)
    {
      printf("mismatch at %d, was: %d, expected: %d\n", i, h_array[i], 1);
      return -1;
    }
  printf("success!\n");
  return 0;
}

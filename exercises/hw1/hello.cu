#include <stdio.h>

__global__ void hello()
{
  auto blockx = blockIdx.x;
  auto blocky = blockIdx.y;
  auto blockz = blockIdx.z;

  auto threadx = threadIdx.x;
  auto thready = threadIdx.y;
  auto threadz = threadIdx.z;

  auto blockDimx = blockDim.x;
  auto blockDimy = blockDim.y;
  auto blockDimz = blockDim.z;

  printf("Hello from blockIdx: (%u, %u, %u), blockDim(%u, %u, %u) threadIdx:( %u, %u, %u)\n",
         blockx, blocky, blockz, blockDimx, blockDimy, blockDimz, threadx, thready, threadz);
}

int main()
{

  dim3 grid(1, 2, 3);  // 12
  dim3 block(1, 2, 3); // 12
  hello<<<grid, block>>>();
  cudaDeviceSynchronize();
}

#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include "include/error.h"
#include "include/timer.h"

const int DSIZE = 32 * 1048576;
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds)
{

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds; idx += gridDim.x * blockDim.x) // a grid-stride loop
    C[idx] = A[idx] + B[idx];                                                                    // do the vector (element) add here
}

int main(int args, char *argv[])
{

  float *h_A, *h_B, *h_C, *h_D, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE]; // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  h_D = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++)
  { // initialize vectors in host memory
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = 0;
    h_D[i] = h_A[i] + h_B[i];
  }

  int blocks = 1;  // modify this line for experimentation
  int threads = 1; // modify this line for experimentation

  for (auto i = 1; i < args; i++)
  {
    if (i == 1)
    {
      blocks = atoi(argv[i]);
    }
    if (i == 2)
    {
      threads = atoi(argv[i]);
    }
  }
  {
    std::stringstream ss;
    ss << "vector_add:(" << blocks << "," << threads << ")";
    auto t = TimeMonitor(ss.str());
    cudaMalloc(&d_A, DSIZE * sizeof(float)); // allocate device space for vector A
    cudaMalloc(&d_B, DSIZE * sizeof(float)); // allocate device space for vector B
    cudaMalloc(&d_C, DSIZE * sizeof(float)); // allocate device space for vector C
    cudaCheckErrors("cudaMalloc failure");   // error checking
    // copy vector A to device:
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    // copy vector B to device:
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    // cuda processing sequence step 1 is complete

    vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");
    // cuda processing sequence step 2 is complete
    //  copy vector C from device to host:
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda processing sequence step 3 is complete
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  }
  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  for (int i = 0; i < DSIZE; i++)
  { // initialize vectors in host memory
    if (h_D[i] != h_C[i])
    {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], h_D[i]);
      return -1;
    }
  }
  return 0;
}

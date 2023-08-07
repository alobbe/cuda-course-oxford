

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid];

    // next, we perform binary tree reduction

    for (int d=blockDim.x/2; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[0] = temp[0];
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, sum;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks   = 1;  // start with only 1 thread block
  num_nthreads = 512;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // compute reference solution

  sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float)) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float),
                              cudaMemcpyDeviceToHost) );

  // check results

  printf("reduction error = %f\n",h_data[0]-sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}

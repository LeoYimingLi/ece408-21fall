// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, int flag) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float scanBlock[BLOCK_SIZE * 2];

  int index = 0;
  int stride = 1;
  if (!flag) {  // flag == 0: scan array
    index = threadIdx.x + (blockIdx.x * blockDim.x * 2);
    stride = blockDim.x;
  } else {  // flag ==1: scan sum
    // actually the minimum sharedmemory size needed here is
    // ceil(len * 1.0 / BLOCK_SIZE) - 1;
    index = (threadIdx.x + 1) * (blockDim.x * 2) - 1;
    stride = blockDim.x * 2;
  }

  int storeIndex = threadIdx.x + (blockIdx.x * blockDim.x * 2);
  
  // load data 
  if (index < len) {
    scanBlock[threadIdx.x] = input[index];
  } else {
    scanBlock[threadIdx.x] = 0;
  }

  if (index + stride < len) {
    scanBlock[threadIdx.x + blockDim.x] = input[index + stride];
  } else {
    scanBlock[threadIdx.x + blockDim.x] = 0;
  }

  //Reduction Step
  for (int stride = 1; stride <= BLOCK_SIZE*2; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if ((index < 2 * BLOCK_SIZE) && ((index - stride) >= 0)) {
      scanBlock[index] += scanBlock[index - stride];
    }
  }

  //Post Scan Step(Distribution Tree)
  for (int stride = 2 * BLOCK_SIZE / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if ((index + stride) < 2 * BLOCK_SIZE) {
      scanBlock[index + stride] += scanBlock[index];
    }
  }

  __syncthreads();
  if (storeIndex < len) {
    output[storeIndex] = scanBlock[threadIdx.x];
  }

  if (storeIndex + blockDim.x < len) {
    output[storeIndex + blockDim.x] = scanBlock[threadIdx.x + blockDim.x];
  }
}


__global__ void add(float *input, float *output, float *sum, int len) {
  __shared__ float increment;

  int index = threadIdx.x + (blockIdx.x * blockDim.x * 2);

  if (threadIdx.x == 0) {  // only the first thread loads the increment
    if (blockIdx.x == 0) {
      increment = 0;
    } else {
      increment = sum[blockIdx.x - 1];
    }
  }
  __syncthreads();

  if (index < len) {
    output[index] = input[index] + increment;
  }
  if (index + blockDim.x < len) {
    output[index + blockDim.x] = input[index + blockDim.x] + increment;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceScanBuffer; // store scan temp results
  float *deviceScanSum; // store scan block sums
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBuffer, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanSum, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(BLOCK_SIZE * 2.0)),   1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  // use deviceOutput to store the temp-value
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceScanBuffer, numElements, 0);
  
  dim3 postScanGrid(1, 1, 1);
  scan<<<postScanGrid, dimBlock>>>(deviceScanBuffer, deviceScanSum, numElements, 1);
  
  add<<<dimGrid, dimBlock>>>(deviceScanBuffer, deviceOutput, deviceScanSum, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceScanBuffer);
  cudaFree(deviceScanSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

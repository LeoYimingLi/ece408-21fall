// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
// Cast the image from float to unsigned char
__global__ void castToUnsignedChar(float *input, unsigned char *output, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = (unsigned char) (255*input[id]); 
  }
}

// Convert the image from RGB to GrayScale
__global__ void castToGray(unsigned char *input, unsigned char *output, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned char r = input[3 * id];
  unsigned char g = input[3 * id + 1];
  unsigned char b = input[3 * id + 2];
  if (id < size) {
    output[id] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}

// Compute the histogram of grayImage
__global__ void computeHistogram(unsigned char *input, unsigned int *output, int size) {
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (idx < size) {
    atomicAdd(&(histo_private[input[idx]]), 1);
    idx += stride;
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[threadIdx.x]), histo_private[threadIdx.x]);
  }
}

// Compute the Cumulative Distribution Function of histogram (scan in only one pass)
__global__ void scan(unsigned int *histogram, float *cdf, int size) {
  __shared__ float scanBlock[HISTOGRAM_LENGTH];

  int i = threadIdx.x;
  if (i < HISTOGRAM_LENGTH) scanBlock[i] = histogram[i];
  if (i + blockDim.x < HISTOGRAM_LENGTH) scanBlock[i+blockDim.x] = histogram[i+blockDim.x];

  // reduction step
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (i + 1) * 2 * stride - 1;
    if (index < HISTOGRAM_LENGTH) {
      scanBlock[index] += scanBlock[index - stride];
    }
  }

  // //Post Scan Step(Distribution Tree)
  for (int stride = ceil(HISTOGRAM_LENGTH / 4.0); stride > 0; stride /= 2) {
    __syncthreads();
    int index = (i + 1) * stride * 2 - 1;
    if(index + stride < HISTOGRAM_LENGTH) {
      scanBlock[index + stride] += scanBlock[index];
    }
  }
  __syncthreads();
  if (i < HISTOGRAM_LENGTH) cdf[i] = ((float) (scanBlock[i] * 1.0) / size);
  if (i + blockDim.x < HISTOGRAM_LENGTH) cdf[i+blockDim.x] = ((float) (scanBlock[i+blockDim.x] * 1.0) / size);
}

// histogram equalization
__global__ void histogramEqualization(unsigned char *ucharImage, float *cdf, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float cdf_min = cdf[0];
  if (id < size) {
    float correct_color = 255.0 * (cdf[ucharImage[id]] - cdf_min) / (1.0 - cdf_min);
    ucharImage[id] = (unsigned char) (min(max(correct_color, 0.0), 255.0));
  }
}

// Cast back to float
__global__ void castToFloat(unsigned char *input, float *output, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = (float) (input[id]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float   *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceImageGrayScale;
  unsigned int  *deviceImageHistogram;
  float   *deviceImageCDF;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  
  // Allocating GPU memory
  cudaMalloc((void **)&deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceImageChar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageGrayScale, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float));
  
  // Copying input memory to the GPU
  cudaMemcpy(deviceImageFloat, hostInputImageData, 
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  
  // Cast the image from float to unsigned char
  dim3 dimGrid1(ceil(imageWidth * imageHeight * imageChannels / 512.0), 1, 1);
  dim3 dimBlock1(512, 1, 1);
  castToUnsignedChar<<<dimGrid1, dimBlock1>>>(deviceImageFloat, deviceImageChar, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  // Convert the image from RGB to GrayScale
  dim3 dimGrid2(ceil(imageWidth * imageHeight / 512.0), 1, 1);
  dim3 dimBlock2(512, 1, 1);
  castToGray<<<dimGrid2, dimBlock2>>>(deviceImageChar, deviceImageGrayScale, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  // Compute the histogram of grayImage
  dim3 dimGrid3(ceil(imageWidth * imageHeight / 256.0), 1, 1);
  dim3 dimBlock3(256, 1, 1);
  computeHistogram<<<dimGrid3, dimBlock3>>>(deviceImageGrayScale, deviceImageHistogram, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  // Compute the Cumulative Distribution Function of histogram (scan)
  dim3 dimGrid4(1, 1, 1);
  dim3 dimBloc4(128, 1, 1);
  scan<<<dimGrid4, dimBloc4>>>(deviceImageHistogram, deviceImageCDF, imageWidth * imageHeight);
  cudaDeviceSynchronize();

  // histogram equalization
  dim3 dimGrid5(ceil(imageWidth * imageHeight * imageChannels/512.0), 1, 1);
  dim3 dimBlock5(512,1,1);
  histogramEqualization<<<dimGrid5, dimBlock5>>>(deviceImageChar, deviceImageCDF, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();

  // Cast back to float
  dim3 dimGrid6(ceil(imageWidth * imageHeight * imageChannels / 512.0), 1, 1);
  dim3 dimBlock6(512,1,1);
  castToFloat<<<dimGrid6, dimBlock6>>>(deviceImageChar, deviceImageFloat, imageWidth * imageHeight * imageChannels);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceImageFloat,
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  // Check Solution 
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  // Freeing GPU Memory 
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageChar);
  cudaFree(deviceImageGrayScale);
  cudaFree(deviceImageHistogram);
  cudaFree(deviceImageCDF);

  // Freeing CPU Memory
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

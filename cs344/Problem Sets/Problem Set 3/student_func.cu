/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>

__global__ 
void reduce_minmax_kernel(float* d_out, const float* const d_in, bool is_min, const size_t size){
  extern __shared__ float sdata[];

  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  if (mid < size){
    sdata[tid] = d_in[mid];
  }
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
    if (tid < s && mid+s < size){
      if (is_min){
        sdata[tid] = min(sdata[tid],  sdata[tid+s]);
      } else{
        sdata[tid] = max(sdata[tid],  sdata[tid+s]);
      }
    }
    __syncthreads();
  }

  if (tid == 0){
    d_out[blockIdx.x] = sdata[tid];
  }

}

int get_block_num(int n, int d){
  return (int)ceil((float)n/d);
}

float reduce_minmax(const float* const h_in, const size_t size, bool is_min){
  int BLOCK_WIDTH = 1024;
  int h_out[1];

  float *d_in, *d_out;
  cudaMalloc((void **)&h_in, sizeof(float)*size);
  cudaMemcpy(d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);

  dim3 thread_dim(BLOCK_WIDTH);
  int shared_memory_size = BLOCK_WIDTH;
  size_t curr_size = size;

  while(curr_size > 1){
    int block_num = get_block_num(curr_size, BLOCK_WIDTH);
    dim3 block_dim(block_num);
    cudaMalloc((void**)&d_out, sizeof(float)*block_num);

    reduce_minmax_kernel<<<block_dim, thread_dim>>>(d_out, d_in, is_min, curr_size);
    cudaDeviceSynchronize();

    cudaFree(d_in);

    d_in = d_out;
    curr_size = block_num;

  }

  cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  return *h_out;


}

// bin = (lum[i] - lumMin) / lumRange * numBins
__global__
void histogram_kernel(unsigned int *d_out, const float const *d_in, int size_bin, int size_in, float max_logLum, float min_logLum){
  int mid = threadIdx.x + blockDim.x*blockIdx.x;
  if (mid >= size_in){
    return;
  }
  float range = max_logLum - min_logLum;

  int bin = (d_in[mid] - min_logLum)/ range* size_bin;

  atomicAdd(&d_out[bin], 1);
}

__global__
void scan_kernel(unsigned int *d_bins, int num_bin){
  int mid = threadIdx.x + blockDim.x*blockIdx.x;
  if (mid >= num_bin){
    return;
  }

  for (int i = 1; i < num_bin; i <<= 1){
    if (mid >= i){
      unsigned int temp = d_bins[mid-i];
      __syncthreads();
      d_bins[mid] += temp;
      __syncthreads();
    }
  }

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  
  dim3 thread_dim(1024);
  dim3 block_dim(get_block_num(numRows*numCols, thread_dim.x));

  min_logLum = reduce_minmax(d_logLuminance, numRows*numCols, true);
  max_logLum = reduce_minmax(d_logLuminance, numRows*numCols, false);
  cudaMemset(d_cdf, 0, sizeof(unsigned int)*numBins);
  histogram_kernel<<<block_dim, thread_dim>>>(d_cdf, d_logLuminance, numBins, numRows*numCols, max_logLum, min_logLum);
  cudaDeviceSynchronize();
  scan_kernel<<<block_dim, thread_dim>>>(d_cdf, numBins);
  cudaDeviceSynchronize();



  


}

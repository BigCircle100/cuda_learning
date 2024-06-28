#include <stdio.h>

// local memory，直接在线程函数中定义使用就可以
__global__ void use_local_memory_GPU(float in){
  float f;    // variable "f" is in local memory and private to each thread
  f = in;     // parameter "in" is in local memory and private to each thread
  // .....
  printf("local res: %f\n", f);
}

// global memory
__global__ void use_global_memory_GPU(float *array){
  // "array" is a pointer into global memory on the device
  array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

// // shared memory
// __global__ void use_shared_memory_GPU(float *array){
//   int index = threadIdx.x;
//   float average, sum = 0.0f;

//   __shared__ float sh_arr[128];
//   sh_arr[index] = array[index];
//   __syncthreads();

//   for(int i = 0; i < index; i++){
//     sum += sh_arr[i];
//   }
//   average = sum / (index*1.0f);

//   if(array[index] > average){
//     array[index] = average;
//   }
//   __syncthreads();
// }


int main(int argc, char** argv){
  // local memory
  use_local_memory_GPU<<<1, 128>>>(2.0f);

  // global memory
  float h_arr[128];
  float *d_arr;

  cudaMalloc((void**) &d_arr, sizeof(float)*128);
  cudaMemcpy((void*)d_arr, (void*)h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);
  use_global_memory_GPU<<<1, 128>>>(d_arr);
  cudaMemcpy((void*)h_arr, (void*)d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost);

  printf("global res: ");
  for (int i = 0; i < 128; i++){
    printf("%f ", h_arr[i]);
  }
  printf("\n");

  // shared memory
  // cudaMemcpy((void*)d_arr, (void*)h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);
  // use_shared_memory_GPU<<<1, 128>>>(d_arr);
  // cudaMemcpy((void*)h_arr, (void*)d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost);
  
  // printf("shared res: ");
  // for (int i = 0; i < 128; i++){
  //   printf("%f ", h_arr[i]);
  // }
  // printf("\n");

}

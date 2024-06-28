#include <stdio.h>
#include <cuda_runtime.h>

// 要实现的功能是2^20个元素求和
// 可以分为两个阶段去做：
// 1. 1024个block，每个block有1024个线程，先对每个block内的数据求和
// 2. 对所有block的求和结果求和

// 这里是第一步：
// 输入d_in[1024*1024]，输出d_out[1024]
// blockDim.x = 1024
// 先分成两份，前512个元素和对应后512个元素相加，结果保留在前512个元素。同步等待所有操作完成，之后s右移，相当于又分成了前后两组
// s一直右移，最后一次就变成了前1个元素和后一个元素
// 最终当前block的结果就是当前block的第一个线程id中的值
__global__ void global_reduce_kernel(int *d_out, int *d_in) {
  // 这里需要注意，当前block中的线程id的计算结果需要保存到总id对应位置的存储中
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  for(unsigned int s = blockDim.x/2 ; s > 0; s >>= 1){
    if (tid < s){
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }

  if (tid == 0){
    d_out[blockIdx.x] = d_in[myId];
  }
}

__global__ void shared_reduce_kernel(int *d_out, int *d_in){
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // 注意共享内存的申请方式，因为这个大小不是确定的
  extern __shared__ int sh_in[];

  sh_in[tid] = d_in[myId];
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
    if (tid < s){
      sh_in[tid] += sh_in[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0){
    d_out[blockIdx.x] = sh_in[tid];
  }
  
}


int main() {
    // 定义 CUDA 事件变量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    bool is_shared = true;

    int h_in[1024*1024];
    int h_out[1024];
    
    for (int i = 0; i < 1024*1024; i++){
      h_in[i] = 1;
    }
    int *d_in, *d_out;

    cudaMalloc((void**)&d_in, sizeof(int)*1024*1024);
    cudaMalloc((void**)&d_out, sizeof(int)*1024);

    cudaMemcpy(d_in, h_in, sizeof(int)*1024*1024, cudaMemcpyHostToDevice);


    // 启动计时
    cudaEventRecord(start);

    if (is_shared){
      shared_reduce_kernel<<<1024, 1024, 1024*sizeof(int)>>>(d_out, d_in);
    }else{
      global_reduce_kernel<<<1024, 1024>>>(d_out, d_in);

    }

    // 停止计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("global 执行时间: %f ms\n", milliseconds);
 
    cudaMemcpy(h_out, d_out, sizeof(int)*1024, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1024; i++){
      printf("%d ", h_out[i]);
    }
    printf("\n");

    // 释放 CUDA 事件变量
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

#include <stdio.h>
#include <unistd.h>

__global__
void add_reduce1(int* d_out, int* d_in, size_t size){
  extern int __shared__ sdata[];

  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  if (mid < size){
    sdata[tid] = d_in[mid];
  }
  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
    if (tid < s && mid+s < size){
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0){
    d_out[blockIdx.x] = sdata[0];
  }
  
}

__global__
void add_reduce2(int* d_out, int* d_in, size_t size){
  extern int __shared__ sdata[];

  int mid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  if (mid < size){
    sdata[tid] = d_in[mid];
  }else{
    sdata[tid] = 0;
  }

  __syncthreads();

  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
    if (tid < s){
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0){
    d_out[blockIdx.x] = sdata[0];
  }
  
}

int get_block_num(int n, int d){
  return (int)ceil((float)n/d);
}

int adding(int* h_in, size_t size){
  int BLOCK_WIDTH = 1024;

  int h_out[1];

  int *d_in, *d_out;
  cudaMalloc((void**)&d_in, sizeof(int)*size);
  cudaMemcpy(d_in, h_in, sizeof(int)*size, cudaMemcpyHostToDevice);

  dim3 thread_dim(BLOCK_WIDTH);
  int shared_mem_size = BLOCK_WIDTH;
  size_t curr_size = size;

  while(curr_size > 1){
    int block_num = get_block_num(curr_size, BLOCK_WIDTH);
    dim3 block_dim(block_num);
    cudaMalloc((void**)&d_out, sizeof(int)*block_num);

    // add_reduce1<<<block_dim, thread_dim, sizeof(int)*shared_mem_size>>>(d_out, d_in, curr_size);
    add_reduce2<<<block_dim, thread_dim, sizeof(int)*shared_mem_size>>>(d_out, d_in, curr_size);
    cudaDeviceSynchronize();

    cudaFree(d_in);

#if DEBUG
    int *temp_out = (int*)malloc(sizeof(int)*block_num);
    cudaMemcpy(temp_out, d_out, sizeof(int)*block_num, cudaMemcpyDeviceToHost);

    printf("curr_size: %zu\n", curr_size);
    printf("block_num: %d\n", block_num);
    sleep(1);
    for (int i = 0; i < block_num; i++){
      printf("%d ", temp_out[i]);
    }
    printf("\n==========\n");
#endif
    
    d_in = d_out;
    curr_size = block_num;

  }

  cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

  return *h_out;

}

int main(int argc, char** argv){
  int row = 2033;
  int col = 29321;
  int value = 1;

  int* h_in = (int*)malloc(row*col*sizeof(int));

  for (int i = 0; i < row*col; i++){
    h_in[i] = value;
  }

  int res = adding(h_in, row*col);

  printf("res: %d\n", res);

  int gt = (int)row*(int)col*value;

  printf("gt: %d\n", gt);

}
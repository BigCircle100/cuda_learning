#include <stdio.h>
#include "gputimer.h"

const int N = 1024;
const int K = 32;

void fill_matrix(int matrix[], int num){
  for (int i = 0; i < num*num; ++i){
    matrix[i] = i;
  }
}

void print_matrix(int matrix[], int num){
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      printf("%d ", matrix[j+i*N]);
    }
    printf("\n");
  }
}

bool compare_matrices(int a[], int b[]){
  for (int i = 0; i < N*N; ++i){
    if (a[i] != b[i]){
      return false;
    }
  }
  return true;
}

// cpu version -- ground truth
void transpose_cpu(int in[], int out[]){
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      out[i + j*N] = in[j + i*N];
    }
  }
}

// gpu serial version
__global__
void transpose_serial(int in[], int out[]){
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      out[i + j*N] = in[j + i*N];
    }
  }
}

// one thread per row
__global__
void transpose_parallel_per_row(int in[], int out[]){
  int i = threadIdx.x;

  for (int j = 0; j < N; ++j){
    out[i + j*N] = in[j + i*N];
  }
}

// one thread per element, which is sub block(K*K)
__global__
void transpose_parallel_per_element(int in[], int out[]){
  int j = blockIdx.x*K + threadIdx.x;
  int i = blockIdx.y*K + threadIdx.y;

  out[i + j*N] = in[j + i*N];
}

// one thread per element, using shared memory
// y corresponds to row, x corresponds to col
__global__
void transpose_parallel_shared(int in[], int out[]){
  __shared__ int tile[K][K];
  
  int x = blockIdx.x * K + threadIdx.x;
  int y = blockIdx.y * K + threadIdx.y;
  
  if (x < N && y < N) {
      tile[threadIdx.y][threadIdx.x] = in[y * N + x];
  }
  
  __syncthreads();
  
  // 转置块偏移
  x = blockIdx.y * K + threadIdx.x;
  y = blockIdx.x * K + threadIdx.y;
  
  if (x < N && y < N) {
      out[y * N + x] = tile[threadIdx.x][threadIdx.y];
  }
}


int main(){
  int numbytes = N*N*sizeof(int);
  int *h_in = (int*)malloc(numbytes);
  int *h_out = (int*)malloc(numbytes);
  int *gt = (int*)malloc(numbytes);

  fill_matrix(h_in, N);
  // printf("input matrix:\n");
  // print_matrix(in, N);
  transpose_cpu(h_in, gt);
  // printf("output matrix:\n");
  // print_matrix(gt, N);

  int *d_in, *d_out;
  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);
  cudaMemcpy(d_in, h_in, numbytes, cudaMemcpyHostToDevice);
  
  GpuTimer timer;

// serial 
  timer.Start();
  transpose_serial<<<1, 1>>>(d_in, d_out);
  timer.Stop();

  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);

  printf("transpose_serial: %g ms\nresult is %s\n", 
          timer.Elapsed(), compare_matrices(gt, h_out)? "correct": "wrong");

// parallel_per_row
  timer.Start();
  transpose_parallel_per_row<<<1, N>>>(d_in, d_out);
  timer.Stop();

  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);

  printf("transpose_parallel_per_row: %g ms\nresult is %s\n", 
          timer.Elapsed(), compare_matrices(gt, h_out)? "correct": "wrong");

// parallel per element
  dim3 blocks(N/K, N/K); // 这里已经确定是整除的了，所以没有进行其他操作
  dim3 threads(K, K);

  timer.Start();
  transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);
  timer.Stop();

  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);

  printf("transpose_parallel_per_element: %g ms\nresult is %s\n", 
          timer.Elapsed(), compare_matrices(gt, h_out)? "correct": "wrong");

// parallel per element with shared memory
  // dim3 blocks(N/K, N/K); // 这里已经确定是整除的了，所以没有进行其他操作
  // dim3 threads(K, K);

  timer.Start();
  transpose_parallel_shared<<<blocks, threads>>>(d_in, d_out);
  timer.Stop();

  cudaMemcpy(h_out, d_out, numbytes, cudaMemcpyDeviceToHost);

  printf("transpose_parallel_shared: %g ms\nresult is %s\n", 
          timer.Elapsed(), compare_matrices(gt, h_out)? "correct": "wrong");

}
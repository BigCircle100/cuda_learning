#include <iostream>
#include <time.h>

// A(M*N) * B(N*K) = C(M*K)

__global__ void matrix_multi_global(const int *A, const int *B, int *C, const int M, const int N, const int K){
  int col = blockDim.x*blockIdx.x + threadIdx.x;
  int row = blockDim.y*blockIdx.y + threadIdx.y;

  if (col >= K || row >= M){
    return;
  }
  int res = 0;

  for (int n = 0; n < N; n++){
    res += A[n+row*N]*B[col+n*K];
  }

  int index = row*K+col;
  C[index] = res;

}

void print_matrix(const int *matrix, const int row, const int col){
  std::cout << "[";
  for (int i = 0; i < row*col; i++){
    if (i%col == 0)
      std::cout << "[";
    std::cout << matrix[i] << ",";
    if (i%col == col-1)
      std::cout << "],";
  }
  std::cout << "]" << std::endl;
  
}

int main(){

  srand(time(NULL));
  int m = 3, n = 4, k = 2;
  int h_a[m*n];
  int h_b[n*k];
  int h_c[m*k];

  for (int i = 0; i < m*n; i++){
    h_a[i] = rand() % 10;
  }
  for (int i = 0; i < n*k; i++){
    h_b[i] = rand() % 10;
  }

  std::cout << "a:" << std::endl;
  print_matrix(h_a, m, n);
  std::cout << "b:" << std::endl;
  print_matrix(h_b, n, k);

  int *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, sizeof(int)*m*n);
  cudaMalloc((void**)&d_b, sizeof(int)*n*k);
  cudaMalloc((void**)&d_c, sizeof(int)*m*k);

  cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

  int blockWidth = 2;
  dim3 blockSize(blockWidth, blockWidth);
  dim3 gridSize(k/blockWidth+1, m/blockWidth+1);
  matrix_multi_global<<<gridSize, blockSize>>>(d_a, d_b, d_c, m, n, k);

  cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
  std::cout << "c: " << std::endl;
  print_matrix(h_c, m, k);
  
}
#include <stdio.h>
#include <iostream>
#include <algorithm>
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


int get_block_num(int n, int d) {
    return (int)ceil( (float)n/(float)d );
}


__global__
void histogram_kernel(const int pass, 
                      int * d_bins,
                      const int * const d_in,
                      const int size){
  int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if (mid >= size)
    return;
  int one = 1;
  int bin = (d_in[mid] & (one << pass)) == (one << pass);
  if (bin)
    atomicAdd(&d_bins[1], 1);
  else
    atomicAdd(&d_bins[0], 1);
}

__global__
void exclusive_scan_kernel(const int pass,
                            const int *const d_in,
                            int *d_scan,
                            const int size,
                            int block_idx){
  int mid = threadIdx.x + blockDim.x * block_idx;
  if (mid >= size)
    return;
  int val = 0;
  int one = 1;

  if (mid > 0)
    val = ((d_in[mid-1] & (one << pass)) == (one << pass)) ? 1 : 0;
  
  d_scan[mid] = val;
  __syncthreads();

  for (int s = 1; s <= size; s *=2){
    if (mid - s >= blockDim.x * block_idx){
      val = d_scan[mid-s];
    }
    __syncthreads();
    if (mid - s >= blockDim.x * block_idx){
      d_scan[mid] += val;
    }
    __syncthreads();
  }


  if (block_idx > 0)
    d_scan[mid] += d_scan[block_idx*blockDim.x - 1];
}

__global__
void move_kernel( const int pass,
                  const int *const d_in,
                  int *d_out,
                  int *d_scan,
                  int one_pos,
                  const int size){
  int mid = threadIdx.x + blockDim.x * blockIdx.x;
  if (mid >= size)
    return;
  int scan = 0;
  int base = 0;
  int one = 1;
  if ((d_in[mid] & (one << pass)) == (one << pass)){
    base = one_pos;
    scan = d_scan[mid];
  }else{
    base = 0;
    scan = mid - d_scan[mid];
  }
  d_out[base+scan] = d_in[mid];

}

int main(){
  int length = 10;
  int h_in[length];
  int h_out[length];
  int h_bins[2];
  for (int i = 0; i < length; i++){
    h_in[i] = rand();
    // h_in[i] = i;
  }

  printf("h_in: \n");
  for (int i = 0; i < length; i++){
    printf("%d ", h_in[i]);
  }
  printf("\n");

  int *d_in, *d_out, *d_scan, *d_bins;
  cudaMalloc((void**)&d_in, sizeof(int)*length);
  cudaMalloc((void**)&d_out, sizeof(int)*length);
  cudaMalloc((void**)&d_scan, sizeof(int)*length);
  cudaMalloc((void**)&d_bins, sizeof(int)*2);

  cudaMemcpy(d_in, h_in, sizeof(int)*length, cudaMemcpyHostToDevice);


  int block_width = 3;
  dim3 thread_dim(block_width);
  dim3 block_dim(get_block_num(length, block_width));


  for (int pass = 0; pass < 32; ++pass){
    cudaMemset(d_bins, 0, 2*sizeof(int));
    cudaMemset(d_scan, 0, length*sizeof(int));
    histogram_kernel<<<block_dim, thread_dim>>>(pass, d_bins, d_in, length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_bins, d_bins, 2*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < get_block_num(length, block_width); ++i){
      exclusive_scan_kernel<<<dim3(1), thread_dim>>>(pass, d_in, d_scan, length, i);
      cudaDeviceSynchronize();
    }
    move_kernel<<<block_dim, thread_dim>>>(pass, d_in, d_out, d_scan, h_bins[0], length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(int)*length, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_in, d_out, sizeof(int)*length, cudaMemcpyDeviceToDevice);
  }


  cudaMemcpy(h_out, d_out, sizeof(int)*length, cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_bins);
  cudaFree(d_scan);

  printf("h_out: \n");
  for (int i = 0; i < length; i++){
    printf("%d ", h_out[i]);
  }
  printf("\n");

  std::sort(h_in, h_in+length);
  bool is_equal = true;
  for (int i = 0; i < length; i++){
    if (h_in[i] != h_out[i]){
      is_equal = false;
      break;
    }
  }
  if (is_equal){
    printf("correct anwser\n");
  }else{
    printf("wrong anwser\n");
  }

}
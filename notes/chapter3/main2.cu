#include <stdio.h>

#define DEBUG 0

int get_block_num(int n, int d){
  return (int)ceil(float(n)/d);
}

// 求直方图
// bin = (value[i] - minvalue) / bin_size
__global__
void histogram_kernel(int *d_out, int *d_in, int size_bin, int size_in, int max_val, int min_val){
  int mid = threadIdx.x + blockDim.x*blockIdx.x;
  if (mid >= size_in){
    return;
  }

  int bin = ((float)d_in[mid]- min_val)/size_bin;

  atomicAdd(&d_out[bin], 1);
}

// 输出d_out是累积分布函数
// hillis + steele
// value[j] = value[j] + (value[j-2^i]>0? value[j-2^i]: 0)
__global__
void scan_kernel(int *d_bins, int num_bin){
  int mid = threadIdx.x + blockDim.x*blockIdx.x;
  if (mid >= num_bin){
    return;
  }

  for (int i = 1; i < num_bin; i*=2){
    if (mid >= i){
      int temp = d_bins[mid-i];
      __syncthreads();
      d_bins[mid] += temp;
      __syncthreads();
    }
  }
    printf("%d \n", d_bins[mid]);
  
}

int main(){
  int size = 1900;
  int size_bin = 2;
  int min_val = 0, max_val = 9;
  int range = max_val-min_val;
  int num_bin = (float) range/size_bin+1;
  int h_in[size], h_out[size_bin];

  for (int i = 0; i < size; ++i){
    h_in[i] = rand()%10;
    // printf("%d ", h_in[i]);
  }

  int *d_in, *d_out;

  cudaMalloc((void**)&d_in, sizeof(int)*size);
  cudaMalloc((void**)&d_out, sizeof(int)*num_bin);

  cudaMemcpy(d_in, h_in, sizeof(int)*size, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, sizeof(int)*num_bin);
  
  dim3 thread_dim(512);
  dim3 block_dim(get_block_num(size, thread_dim.x));
  histogram_kernel<<<block_dim, thread_dim>>>(d_out, d_in, size_bin, size, max_val, min_val);
  cudaDeviceSynchronize();

#if DEBUG
  cudaMemcpy(h_out, d_out, sizeof(int)*num_bin, cudaMemcpyDeviceToHost);

  printf("\nbins: \n");
  int sum = 0;
  for (int i = 0; i < num_bin; ++i){
    printf("%d ", h_out[i]);
    sum += h_out[i];
  }

  printf("\n%d \n", sum);

  int debug[10];

  for (int i = 0; i < 10; i++){
    debug[i] = 0;
  }
  for (int i = 0; i < size; i++){
    debug[h_in[i]] += 1;
  }

  printf("gt: \n");
  for (int i = 0; i < 10; i++){
    printf("%d ", debug[i]);
  }

  
#endif
  dim3 thread_dim_scan(2);
  dim3 block_dim_scan(get_block_num(num_bin, thread_dim_scan.x));
  scan_kernel<<<block_dim_scan, thread_dim>>>(d_out, num_bin);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_out, sizeof(int)*num_bin, cudaMemcpyDeviceToHost);

  printf("cdf bins: \n");
  for (int i = 0; i < num_bin; ++i){
    printf("%d ", h_out[i]);
  }
}
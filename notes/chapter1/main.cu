#include <stdio.h>

// kernel func
__global__
void square(float *d_out, float *d_in){
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f*f;
}

int main(int argc, char **argv){
  const int ARRAY_SIZE = 64;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  float h_in[ARRAY_SIZE];
  for(int i = 0; i < ARRAY_SIZE; i++){
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];

  float * d_in;
  float * d_out;

  // alloc device mem, use bytes
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  // s2d
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // <<<block, thread>>>
  square<<<1, ARRAY_SIZE>>>(d_out, d_in);

  // d2s
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  for (int i = 0 ; i < ARRAY_SIZE; i++){
    printf("%f", h_out[i]);
    printf(((i%4)!=3)?"\t":"\n");
  }

  // delete
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;

  
}
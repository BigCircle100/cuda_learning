#include <stdio.h>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello(){
  printf("hello %d\n", blockIdx.x);
}

int main(){
  hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();
  cudaDeviceSynchronize();
  printf("finished");

}
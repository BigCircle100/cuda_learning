#include <stdio.h>

__global__ void calculateAverage(int *array, int size, float *result) {
    // 共享内存用于存储每个线程块的部分和
    __shared__ float partialSum[256];

    int tid = threadIdx.x;
    int totalThreads = blockDim.x;

    float sum = 0.0f;

    // 每个线程计算自己负责的部分和
    for (int i = tid; i < size; i += totalThreads) {
        sum += array[i];
    }

    // 将每个线程的部分和存储到共享内存中
    partialSum[tid] = sum;

    // 等待所有线程完成部分和的计算
    __syncthreads();

    // 用一个线程将每个线程块的部分和相加得到总和
    if (tid == 0) {
        float blockSum = 0.0f;
        for (int i = 0; i < totalThreads; i++) {
            blockSum += partialSum[i];
        }
        *result = blockSum / size;
    }
}

int main() {
    int size = 1000;
    int array[size];
    float result;

    // 初始化数组数据
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }

    int *d_array;
    float *d_result;

    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);

    // 调用 kernel 函数计算平均值
    calculateAverage<<<1, 256>>>(d_array, size, d_result);

    // 将结果从设备复制回主机
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Average: %f\n", result);

    // 释放设备内存
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}

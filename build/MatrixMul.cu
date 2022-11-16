// System includes
#include <stdio.h>
#include <sys/time.h>

// CUDA runtime
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess) {                                          \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }

const int TILE_WIDTH = 2;

void initialData(float* ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

void MatrixMulCPU(float* M, float* N, float* P, int width)
{
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float mul_sum = 0.0;
            for (int k = 0; k < width; ++k) {
                mul_sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = mul_sum;
        }
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    for (int i = 0; i < N * N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}

void printMatrix(float* C, const int nx, const int ny)
{
    float* ic = C;
    printf("Matrix<%d,%d>: \n", ny, nx);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6f ", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
}

// d_P中一个点用一个线程进行计算，一个block内正好包含TILE_WIDTH*TILE_WIDTH个线程。
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // 表示d_P中线程的索引
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identity the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Collaborative loading of d_M abd d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
}

int main()
{
    int width = 4; // 矩阵的边长
    int size = width * width; // 矩阵的大小
    float* h_M = (float*)malloc(size * sizeof(float));
    float* h_N = (float*)malloc(size * sizeof(float));
    float* h_P = (float*)malloc(size * sizeof(float));
    initialData(h_M, size);
    initialData(h_N, size);
    printf("M: \n");
    printMatrix(h_M, width, width);
    printf("N: \n");
    printMatrix(h_N, width, width);

    float* d_M = nullptr;
    float* d_N = nullptr;
    float* d_P = nullptr;
    float* from_gpu_P = (float*)malloc(size*sizeof(float));
    CHECK(cudaMalloc((float**)&d_M, size*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_N, size*sizeof(float)));
    CHECK(cudaMalloc((float**)&d_P, size*sizeof(float)));

    CHECK(cudaMemcpy(d_M, h_M, size*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_N, h_N, size*sizeof(float), cudaMemcpyHostToDevice));
    dim3 gridDim(2, 2, 1);
    dim3 blockDim(width / TILE_WIDTH, width / TILE_WIDTH, 1);
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, width);
    cudaDeviceSynchronize();
    
    cudaMemcpy(from_gpu_P, d_P, size*sizeof(float), cudaMemcpyDeviceToHost);
    MatrixMulCPU(h_M, h_N, h_P, width);
    printf("CPU MATRIX: \n");
    printMatrix(h_P, width, width);
    printf("GPU MATRIX: \n");
    printMatrix(from_gpu_P, width, width);
    checkResult(h_P, from_gpu_P, width);

    free(h_M);
    free(h_N);
    free(h_P);
    free(from_gpu_P);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

}

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<N) {
        C[i] = A[i]+B[i];
    }
}

void vector_add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    int N = A.numel();
    int threads=256;
    int blocks=(N+threads-1)/threads;
    
    vector_add_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
}
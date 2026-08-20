#include <torch/extension.h>

void vector_add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

void vector_add(torch::Tensor A, torch::Tensor B, torch::Tensor C){
    vector_add_cuda(A, B, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("vector_add", &vector_add, "Vector addition CUDA");
}
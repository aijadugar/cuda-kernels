#include <torch/extension.h>

void sft_forward(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2
);

void sft(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bias_correction1,
    double bias_correction2
){
    sft_forward(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sft", &sft, "SFT optimizer (CUDA)");
}
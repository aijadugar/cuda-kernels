#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_bf16.h>

__global__ void fused_adamw_kernel(
    __nv_bfloat16* __restrict__ param,
    const __nv_bfloat16* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2
){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;

    float g=__bfloat162float(grad[i]);
    float p=__bfloat162float(param[i]);

    p=p-lr*weight_decay*p;

    float m=beta1*exp_avg[i]+(1.0f-beta1)*g;
    float v=beta2*exp_avg_sq[i]+(1.0f-beta2)*g*g;

    exp_avg[i]=m;
    exp_avg_sq[i]=v;

    float m_hat=m/bias_correction1;
    float v_hat=v/bias_correction2;

    float update=lr*m_hat/(sqrtf(v_hat)+eps);
    p=p-update;

    param[i]=__float2bfloat16(p);
}

void fused_adamw_cuda(
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
    int N=param.numel();
    int threads=256;
    int blocks=(N+threads-1)/threads;

    fused_adamw_kernel<<<blocks, threads>>>(
        reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        N,
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(eps),
        static_cast<float>(weight_decay),
        static_cast<float>(bias_correction1),
        static_cast<float>(bias_correction2)
    );
}
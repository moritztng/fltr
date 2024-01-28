#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

cublasHandle_t cublas_handle;

__global__ void dequantize(float *out, int8_t *quants, float *scales, uint16_t group_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = quants[i] * scales[i / group_size];
}

extern "C"
{
    void cuda_init()
    {
        cublasCreate(&cublas_handle);
    }

    void cuda_matmul(float *out, int8_t *a_h_quants, int8_t *b_h_quants, float *a_h_scales, float *b_h_scales, uint32_t a, uint32_t b, uint32_t n, uint16_t group_size)
    {
        int8_t *a_d_quants, *b_d_quants;
        float *out_d_values, *a_d_values, *b_d_values, *a_d_scales, *b_d_scales;
        const uint32_t a_len = a * n, b_len = b * n, out_len = a * b;
        cudaMalloc(&out_d_values, out_len * sizeof(float));
        cudaMalloc(&a_d_values, a_len * sizeof(float));
        cudaMalloc(&b_d_values, b_len * sizeof(float));
        cudaMalloc(&a_d_quants, a_len * sizeof(int8_t));
        cudaMalloc(&b_d_quants, b_len * sizeof(int8_t));
        cudaMalloc(&a_d_scales, a_len / group_size * sizeof(float));
        cudaMalloc(&b_d_scales, b_len / group_size * sizeof(float));
        cudaMemcpy(a_d_quants, a_h_quants, a_len * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d_quants, b_h_quants, b_len * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(a_d_scales, a_h_scales, a_len / group_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d_scales, b_h_scales, b_len / group_size * sizeof(float), cudaMemcpyHostToDevice);
        dequantize<<<a_len / group_size, group_size>>>(a_d_values, a_d_quants, a_d_scales, group_size);
        dequantize<<<b_len / group_size, group_size>>>(b_d_values, b_d_quants, b_d_scales, group_size);
        float alpha = 1, beta = 0;
        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, a, b, n, &alpha, a_d_values, n, b_d_values, n, &beta, out_d_values, a);
        cudaMemcpy(out, out_d_values, out_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(out_d_values);
        cudaFree(a_d_values);
        cudaFree(b_d_values);
        cudaFree(a_d_quants);
        cudaFree(b_d_quants);
        cudaFree(a_d_scales);
        cudaFree(b_d_scales);
    }
}

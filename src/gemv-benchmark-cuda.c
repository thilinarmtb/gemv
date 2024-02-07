#include "gemv-benchmark-impl.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

static cublasHandle_t handle = NULL;
static float *d_A = NULL;
static float *d_x = NULL;
static float *d_y = NULL;
static int n = 0;

static void init(int device, int n_, const float *A, const float *x) {
  cudaSetDevice(device);
  n = n_;

  cudaMalloc((void **)&d_A, n * n * sizeof(float));
  assert(d_A != NULL);
  cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_x, n * sizeof(float));
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  assert(d_x != NULL);

  cudaMalloc((void **)&d_y, n * sizeof(float));
  assert(d_y != NULL);

  cublasCreate(&handle);
}

static void benchmark(int num_repeats, float *y) {
  float alpha = 1.0f, beta = 0.0f;
  for (int i = 0; i < num_repeats; ++i) {
    cublasSgemv(handle, CUBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y,
                1);
  }
  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

static void finalize(void) {
  cudaFree(d_A), d_A = NULL;
  cudaFree(d_x), d_x = NULL;
  cudaFree(d_y), d_y = NULL;
  cublasDestroy(handle), handle = NULL;
}

void gemv_benchmark_register_cuda(void) {
  gemv_benchmark_register_backend("cuda", init, benchmark, finalize);
}

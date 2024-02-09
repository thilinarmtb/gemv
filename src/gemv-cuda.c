#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemv-impl.h"

static inline void check_cuda_runtime_(cudaError_t err, const char *file,
                                       const unsigned line) {
  if (err == cudaSuccess)
    return;
  fprintf(stderr, "CUDA runtime error: %s in file: %s line: %u\n",
          cudaGetErrorString(err), file, line);
  exit(EXIT_FAILURE);
}

#define check_cuda_runtime(call) check_cuda_runtime_((call), __FILE__, __LINE__)

static cublasHandle_t handle = NULL;
static float *d_A = NULL, *d_x = NULL, *d_y = NULL;
static int n = 0;

static void init(int device, int n_, const float *A, const float *x) {
  check_cuda_runtime(cudaSetDevice(device));
  n = n_;

  check_cuda_runtime(cudaMalloc((void **)&d_A, n * n * sizeof(float)));
  check_cuda_runtime(
      cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice));

  check_cuda_runtime(cudaMalloc((void **)&d_x, n * sizeof(float)));
  check_cuda_runtime(
      cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

  check_cuda_runtime(cudaMalloc((void **)&d_y, n * sizeof(float)));

  cublasCreate(&handle);
}

static void benchmark(int num_repeats, float *y) {
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemv(handle, CUBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);
  check_cuda_runtime(
      cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToDevice));
}

static void finalize(void) {
  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
  check_cuda_runtime(cudaFree(d_x)), d_x = NULL;
  check_cuda_runtime(cudaFree(d_y)), d_y = NULL;
  cublasDestroy(handle), handle = NULL;
}

void gemv_register_cuda(void) {
  gemv_register_backend("cuda", init, benchmark, finalize);
}

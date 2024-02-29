#include "backends/gemv-backend-cuda.h"

static float *d_A = NULL;
static int n = 0;

static void cuda_init(int device, int n_, const float *A) {
  check_cuda_runtime(cudaSetDevice(device));
  n = n_;

  check_cuda_runtime(cudaMalloc((void **)&d_A, n * n * sizeof(float)));
  check_cuda_runtime(
      cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice));
}

static void cuda_gemv(float *d_y, const float *d_x) {}

static void cuda_finalize(void) {
  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
}

void gemv_register_cuda(void) {
  gemv_register_backend("cuda", cuda_init, cuda_copy, cuda_gemv, cuda_finalize);
}

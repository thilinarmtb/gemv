#include "gemv-cuda.h"

#include <cublas_v2.h>

static inline void check_cublas_(cublasStatus_t status, const char *file,
                                 const unsigned line) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  fprintf(stderr, "cuBLAS error: %d in file: %s line: %u\n", status, file,
          line);
  exit(EXIT_FAILURE);
}

#define check_cublas(call) check_cublas_(call, __FILE__, __LINE__)

static cublasHandle_t handle = NULL;
static float *d_A = NULL;
static int n = 0;

static void cublas_init(int device, int n_, const float *A) {
  check_cuda_runtime(cudaSetDevice(device));
  n = n_;

  check_cuda_runtime(cudaMalloc((void **)&d_A, n * n * sizeof(float)));
  check_cuda_runtime(
      cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice));

  cublasCreate(&handle);
}

static void cublas_gemv(float *d_y, const float *d_x) {
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemv(handle, CUBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);
}

static void cublas_finalize(void) {
  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
  cublasDestroy(handle), handle = NULL;
}

void gemv_register_cublas(void) {
  gemv_register_backend("cublas", cublas_init, cublas_copy, cublas_gemv,
                        cublas_finalize);
}

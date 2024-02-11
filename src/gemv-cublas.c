#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gemv-impl.h"

static inline void check_cuda_runtime_(cudaError_t err, const char *file,
                                       const unsigned line) {
  if (err == cudaSuccess) return;
  fprintf(stderr, "CUDA runtime error: %s in file: %s line: %u\n",
          cudaGetErrorString(err), file, line);
  exit(EXIT_FAILURE);
}

#define check_cuda_runtime(call) check_cuda_runtime_((call), __FILE__, __LINE__)

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

static void cublas_copy(void *dest, const void *src, size_t count,
                        gemv_direction_t direction) {
  enum cudaMemcpyKind kind = cudaMemcpyDefault;
  switch (direction) {
  case GEMV_D2H:
    kind = cudaMemcpyDeviceToHost;
    break;
  case GEMV_H2D:
    kind = cudaMemcpyHostToDevice;
    break;
  }

  check_cuda_runtime(cudaMemcpy(dest, src, count, kind));
}

static void cublas_finalize(void) {
  check_cuda_runtime(cudaFree(d_A)), d_A = NULL;
  cublasDestroy(handle), handle = NULL;
}

void gemv_register_cuda(void) {
  gemv_register_backend("cublas", cublas_init, cublas_copy, cublas_gemv,
                        cublas_finalize);
}

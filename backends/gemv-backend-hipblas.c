#include "backends/gemv-backend-hip.h"

#include <hipblas/hipblas.h>

static inline void check_hipblas_(hipblasStatus_t status, const char *file,
                                  const unsigned line) {
  if (status == HIPBLAS_STATUS_SUCCESS) return;
  fprintf(stderr, "hipBLAS error: %d in file: %s line: %u\n", status, file,
          line);
  exit(EXIT_FAILURE);
}

#define check_hipblas(call) check_hipblas_(call, __FILE__, __LINE__)

static hipblasHandle_t handle = NULL;
static float *d_A = NULL;
static int n = 0;

static void hipblas_init(int device, int n_, const float *A) {
  check_hip_runtime(hipSetDevice(device));
  n = n_;

  check_hip_runtime(hipMalloc((void **)&d_A, n * n * sizeof(float)));
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));

  hipblasCreate(&handle);
}

static void hipblas_gemv(float *d_y, const float *d_x) {
  float alpha = 1.0f, beta = 0.0f;
  hipblasSgemv(handle, HIPBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y,
               1);
}

static void hipblas_finalize(void) {
  check_hip_runtime(hipFree(d_A)), d_A = NULL;
  hipblasDestroy(handle), handle = NULL;
}

void gemv_register_hipblas(void) {
  gemv_backend_register("hipblas", hipblas_init, hip_copy, hipblas_gemv,
                        hipblas_finalize);
}

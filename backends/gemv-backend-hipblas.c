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
static void *d_A = NULL;
static unsigned n = 0, m = 0;
static int initialized = 0;

static void hipblas_init(const struct gemv_t *gemv) {
  check_hip_runtime(hipSetDevice(gemv->device));

  n = gemv->n, m = gemv->m;
  check_hip_runtime(hipMalloc((void **)&d_A, n * m * sizeof(double)));

#if 0
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));
#endif

  hipblasCreate(&handle);

  initialized = 1;
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

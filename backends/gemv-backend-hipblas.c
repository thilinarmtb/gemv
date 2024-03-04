#include "backends/gemv-backend-hip.h"

#include <hipblas/hipblas.h>

static inline void check_hipblas_(hipblasStatus_t status, const char *file,
                                  const unsigned line) {
  char *error = NULL;
  switch (status) {
  case HIPBLAS_STATUS_SUCCESS: return; break;
#define add_case(A)                                                            \
  case A: error = #A; break
    add_case(HIPBLAS_STATUS_NOT_INITIALIZED);
    add_case(HIPBLAS_STATUS_ALLOC_FAILED);
    add_case(HIPBLAS_STATUS_INVALID_VALUE);
    add_case(HIPBLAS_STATUS_MAPPING_ERROR);
    add_case(HIPBLAS_STATUS_EXECUTION_FAILED);
    add_case(HIPBLAS_STATUS_INTERNAL_ERROR);
    add_case(HIPBLAS_STATUS_NOT_SUPPORTED);
    add_case(HIPBLAS_STATUS_ARCH_MISMATCH);
    add_case(HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
    add_case(HIPBLAS_STATUS_INVALID_ENUM);
    add_case(HIPBLAS_STATUS_UNKNOWN);
#undef add_case
  default: break;
  }

  fprintf(stderr, "hipBLAS error: %s in file: \"%s\" line: %u\n", error, file,
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

  check_hipblas(hipblasCreate(&handle));

  initialized = 1;
}

static void hipblas_gemv(float *d_y, const float *d_x) {
  float alpha = 1.0f, beta = 0.0f;
  check_hipblas(hipblasSgemv(handle, HIPBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1,
                             &beta, d_y, 1));
}

static void hipblas_finalize(void) {
  check_hip_runtime(hipFree(d_A)), d_A = NULL;
  check_hipblas(hipblasDestroy(handle)), handle = NULL;
}

void gemv_register_hipblas(void) {
  gemv_backend_register("hipblas", hipblas_init, hip_copy, hipblas_gemv,
                        hipblas_finalize);
}

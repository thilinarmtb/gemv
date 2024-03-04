#include "backends/gemv-backend-hip.h"

#include <hipblas/hipblas.h>

static inline void check_hipblas_(hipblasStatus_t status, const char *file,
                                  const unsigned line) {
  char *error = NULL;

#define add_case(A)                                                            \
  case A: error = #A; break

  // clang-format off
  switch (status) {
  case HIPBLAS_STATUS_SUCCESS: return; break;
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
  default: break;
  }
  // clang-format on

#undef add_case

  gemv_log(GEMV_ERROR, "hipBLAS error: %s in file: \"%s\" line: %u", error,
           file, line);
}

#define check_hipblas(call) check_hipblas_(call, __FILE__, __LINE__)

static hipblasHandle_t handle = NULL;
static void *d_A = NULL;
static int initialized = 0;

static void hipblas_init(const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "hipblas_init: ...", initialized);
  if (initialized) return;

  check_hip_runtime(hipSetDevice(gemv->device));

  const unsigned n = gemv->n, m = gemv->m;
  check_hip_runtime(hipMalloc((void **)&d_A, n * m * sizeof(double)));

  size_t unit_size = gemv_unit_size(gemv->precision);
  void *A = gemv_malloc(m * n * unit_size);
  gemv_convert(A, gemv->A, n * m, gemv->precision);

  check_hip_runtime(
      hipMemcpy(d_A, A, n * m * unit_size, hipMemcpyHostToDevice));

  gemv_free(&A);

  check_hipblas(hipblasCreate(&handle));

  initialized = 1;
  gemv_log(GEMV_INFO, "hipblas_init: done.");
}

static void hipblas_gemv(void *d_y, const void *d_x,
                         const struct gemv_t *gemv) {
  float alpha_f = 1.0f, beta_f = 0.0f;
  double alpha_d = 1.0, beta_d = 0.0;
  switch (gemv->precision) {
  case GEMV_FP32:
    check_hipblas(hipblasSgemv(handle, HIPBLAS_OP_T, gemv->m, gemv->n, &alpha_f,
                               d_A, gemv->n, d_x, 1, &beta_f, d_y, 1));
    break;
  case GEMV_FP64:
    check_hipblas(hipblasDgemv(handle, HIPBLAS_OP_T, gemv->m, gemv->n, &alpha_d,
                               d_A, gemv->n, d_x, 1, &beta_d, d_y, 1));
    break;
  default: break;
  }
}

static void hipblas_finalize(void) {
  gemv_log(GEMV_INFO, "hipblas_finalize: ...", initialized);
  if (!initialized) return;

  check_hip_runtime(hipFree(d_A)), d_A = NULL;
  check_hipblas(hipblasDestroy(handle)), handle = NULL;
  initialized = 0;

  gemv_log(GEMV_INFO, "hipblas_finalize: done.");
}

void gemv_register_hipblas(void) {
  gemv_backend_register("hipblas", hipblas_init, hip_copy, hipblas_gemv,
                        hipblas_finalize);
}

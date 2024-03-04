#include "backends/gemv-backend-hip.h"

static void *d_A = NULL;
static unsigned n = 0, m = 0;
static int initialized = 0;

static void hip_init(const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "hip_init: initialized = %d", initialized);
  if (initialized) return;

  check_hip_runtime(hipSetDevice(gemv->device));
  n = gemv->n, m = gemv->m;
  check_hip_runtime(hipMalloc((void **)&d_A, n * m * sizeof(double)));

  size_t unit_size = gemv_unit_size(gemv->precision);
  void *A = gemv_malloc(m * n * unit_size);
  gemv_convert(A, gemv->A, n * m, gemv->precision);

  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));

  gemv_free(&A);

  initialized = 1;
  gemv_log(GEMV_INFO, "hip_init: done.");
}

static void hip_gemv(float *d_y, const float *d_x) {}

static void hip_finalize(void) {
  gemv_log(GEMV_INFO, "hip_finalize: initialized = %d", initialized);
  if (!initialized) return;

  check_hip_runtime(hipFree(d_A)), d_A = NULL;
  initialized = 0;

  gemv_log(GEMV_INFO, "hip_finalize: done.");
}

void gemv_register_hip(void) {
  gemv_backend_register("hip", hip_init, hip_copy, hip_gemv, hip_finalize);
}

#include "backends/gemv-backend-hip.h"

static void *d_A = NULL;
static unsigned n = 0, m = 0;
static int initialized = 0;

static void hip_init(const struct gemv_t *gemv) {
  check_hip_runtime(hipSetDevice(gemv->device));

  n = gemv->n, m = gemv->m;
  check_hip_runtime(hipMalloc((void **)&d_A, n * m * sizeof(double)));

#if 0
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));
#endif

  initialized = 1;
}

static void hip_gemv(float *d_y, const float *d_x) {}

static void hip_finalize(void) { check_hip_runtime(hipFree(d_A)), d_A = NULL; }

void gemv_register_hip(void) {
  gemv_backend_register("hip", hip_init, hip_copy, hip_gemv, hip_finalize);
}

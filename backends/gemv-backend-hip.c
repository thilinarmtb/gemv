#include "backends/gemv-backend-hip.h"

static void *d_A = NULL;
static int initialized = 0;

static void hip_gemv(void *d_y, const void *d_x, const struct gemv_t *gemv) {}

static void hip_finalize(void) {
  gemv_log(GEMV_INFO, "hip_finalize: ...");
  if (!initialized) return;

  check_hip_runtime(hipFree(d_A)), d_A = NULL;
  initialized = 0;

  gemv_log(GEMV_INFO, "hip_finalize: done.");
}

static void hip_init_aux(const struct gemv_t *gemv) {
  check_hip_runtime(hipSetDevice(gemv->device));

  const unsigned m = gemv->m, n = gemv->n;
  check_hip_runtime(hipMalloc((void **)&d_A, m * n * sizeof(double)));

  const size_t unit_size = gemv_unit_size(gemv->precision);
  void *const A = gemv_malloc(m * n * unit_size);
  gemv_convert(A, gemv->A, m * n, gemv->precision);

  check_hip_runtime(
      hipMemcpy(d_A, A, m * n * unit_size, hipMemcpyHostToDevice));

  gemv_free(&A);
}

static void hip_init(struct gemv_backend_t *backend,
                     const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "hip_init: ...");
  if (initialized) return;

  backend->malloc = hip_malloc;
  backend->free = hip_free;
  backend->copy = hip_copy;
  backend->run = hip_gemv;
  backend->finalize = hip_finalize;

  hip_init_aux(gemv);

  initialized = 1;
  gemv_log(GEMV_INFO, "hip_init: done.");
}

void gemv_register_hip(void) { gemv_backend_register("hip", hip_init); }

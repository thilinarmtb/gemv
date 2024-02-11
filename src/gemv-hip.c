#include <hipblas/hipblas.h>

static float *d_A = NULL;
static int n = 0;

static void hip_init(int device, int n_, const float *A) {
  check_hip_runtime(hipSetDevice(device));
  n = n_;

  check_hip_runtime(hipMalloc((void **)&d_A, n * n * sizeof(float)));
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));
}

static void hip_gemv(float *d_y, const float *d_x) {}

static void hip_finalize(void) {
  check_hip_rumtime(hipFree(d_A)), d_A = NULL;
  hipblasDestroy(handle), handle = NULL;
}

void gemv_register_hip(void) {
  gemv_register_backend("hip", hip_init, hip_copy, hip_gemv, hip_finalize);
}

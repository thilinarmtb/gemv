#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "gemv-impl.h"

static inline void check_hip_runtime_(hipError_t err, const char *file,
                                      const unsigned line) {
  if (err == hipSuccess)
    return;
  fprintf(stderr, "HIP runtime error: %s in file: %s line: %u\n",
          hipGetErrorString(err), file, line);
  exit(EXIT_FAILURE);
}

#define check_hip_runtime(call) check_hip_runtime_((call), __FILE__, __LINE__)

static hipblasHandle_t handle = NULL;
static float *d_A = NULL;
static int n = 0;

static void init(int device, int n_, const float *A) {
  check_hip_runtime(hipSetDevice(device));
  n = n_;

  check_hip_runtime(hipMalloc((void **)&d_A, n * n * sizeof(float)));
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));

  hipblasCreate(&handle);
}

static void gemv(float *d_y, const float *d_x) {
  float alpha = 1.0f, beta = 0.0f;
  hipblasSgemv(handle, HIPBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y,
               1);
}

static void copy(void *dest, const void *src, size_t count,
                 gemv_direction_t direction) {
  enum hipMemcpyKind kind = hipMemcpyDefault;
  switch (direction) {
  case GEMV_D2H:
    kind = hipMemcpyDeviceToHost;
    break;
  case GEMV_H2D:
    kind = hipMemcpyHostToDevice;
    break;
  }

  check_hip_runtime(hipMemcpy(dest, src, count, kind));
}

static void finalize(void) {
  check_hip_rumtime(hipFree(d_A)), d_A = NULL;
  hipblasDestroy(handle), handle = NULL;
}

void gemv_register_hip(void) {
  gemv_register_backend("hip", init, copy, gemv, finalize);
}

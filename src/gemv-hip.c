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
static float *d_A = NULL, *d_x = NULL, *d_y = NULL;
static int n = 0;

static void init(int device, int n_, const float *A, const float *x) {
  check_hip_runtime(hipSetDevice(device));
  n = n_;

  check_hip_runtime(hipMalloc((void **)&d_A, n * n * sizeof(float)));
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));

  check_hip_runtime(hipMalloc((void **)&d_x, n * sizeof(float)));
  check_hip_runtime(
      hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice));

  check_hip_runtime(hipMalloc((void **)&d_y, n * sizeof(float)));

  hipblasCreate(&handle);
}

static void benchmark(int num_repeats, float *y) {
  float alpha = 1.0f, beta = 0.0f;
  hipblasSgemv(handle, HIPBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y,
               1);
  check_hip_runtime(
      hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToDevice));
}

static void finalize(void) {
  check_hip_rumtime(hipFree(d_A)), d_A = NULL;
  check_hip_rumtime(hipFree(d_x)), d_x = NULL;
  check_hip_rumtime(hipFree(d_y)), d_y = NULL;
  hipblasDestroy(handle), handle = NULL;
}

void gemv_register_hip(void) {
  gemv_register_backend("hip", init, benchmark, finalize);
}

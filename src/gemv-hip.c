#include "gemv-impl.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

static hipblasHandle_t handle = NULL;
static float *d_A = NULL;
static float *d_x = NULL;
static float *d_y = NULL;
static int n = 0;

static void init(int device, int n_, const float *A, const float *x) {
  hipSetDevice(device);
  n = n_;

  hipMalloc((void **)&d_A, n * n * sizeof(float));
  assert(d_A != NULL);
  hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice);

  hipMalloc((void **)&d_x, n * sizeof(float));
  hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
  assert(d_x != NULL);

  hipMalloc((void **)&d_y, n * sizeof(float));
  assert(d_y != NULL);

  hipblasCreate(&handle);
}

static void benchmark(int num_repeats, float *y) {
  float alpha = 1.0f, beta = 0.0f;
  for (int i = 0; i < num_repeats; ++i) {
    hipblasSgemv(handle, HIPBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y,
                 1);
  }
  hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToDevice);
}

static void finalize(void) {
  hipFree(d_A), d_A = NULL;
  hipFree(d_x), d_x = NULL;
  hipFree(d_y), d_y = NULL;
  hipblasDestroy(handle), handle = NULL;
}

void gemv_register_hip(void) {
  gemv_register_backend("hip", init, benchmark, finalize);
}

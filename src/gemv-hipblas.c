#include <hipblas/hipblas.h>

static hipblasHandle_t handle = NULL;
static float *d_A = NULL;
static int n = 0;

static void hipblas_init(int device, int n_, const float *A) {
  check_hip_runtime(hipSetDevice(device));
  n = n_;

  check_hip_runtime(hipMalloc((void **)&d_A, n * n * sizeof(float)));
  check_hip_runtime(
      hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice));

  hipblasCreate(&handle);
}

static void hipblas_gemv(float *d_y, const float *d_x) {
  float alpha = 1.0f, beta = 0.0f;
  hipblasSgemv(handle, HIPBLAS_OP_T, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y,
               1);
}

static void hipblas_finalize(void) {
  check_hip_rumtime(hipFree(d_A)), d_A = NULL;
  hipblasDestroy(handle), handle = NULL;
}

void gemv_register_hip(void) {
  gemv_register_backend("hipblas", hipblas_init, hip_copy, hipblas_gemv,
                        hipblas_finalize);
}

#include "backends/gemv-backend-hip.h"

#include <hip/hiprtc.h>

static void *d_A = NULL;
static int initialized = 0;

static hipModule_t module = NULL;
static hipFunction_t kernel = NULL;

const char *gemv_kernel_0 =
    "extern \"C\" __global__ void gemv(%s *y, const %s *A, const %s *x, const "
    "unsigned m, "
    "const unsigned n) {\n"
    "  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (idx < m) {\n"
    "    %s dot = 0;\n"
    "    for (int i = 0; i < n; i++)\n"
    "      dot += A[idx * n + i] * x[i];\n"
    "    y[idx] = dot;\n"
    "  }\n"
    "}\n";

static void hip_run(void *d_y, const void *d_x, const struct gemv_t *gemv) {
  if (!initialized)
    gemv_log(GEMV_ERROR, "hip_run: HIP backend is not initialized !");

  gemv_log(GEMV_INFO, "y = %p, x = %p, m = %u, n = %u", d_y, d_x, gemv->m,
           gemv->n);

  void *arguments[] = {&d_y, &d_A, &d_x, (void *)&gemv->m, (void *)&gemv->n};
  hipModuleLaunchKernel(kernel, (gemv->m + 31) / 32, 1, 1, 32, 1, 1, 0, 0,
                        arguments, NULL);
}

static void hip_finalize(void) {
  gemv_log(GEMV_INFO, "hip_finalize: ...");
  if (!initialized) return;

  check_hip_runtime(hipFree(d_A)), d_A = NULL;
  hipModuleUnload(module), module = NULL, kernel = NULL;
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

  char source[BUFSIZ];
  const char *precision = gemv_precision_to_str(gemv->precision);
  snprintf(source, BUFSIZ, gemv_kernel_0, precision, precision, precision,
           precision);
  gemv_log(GEMV_INFO, "hip_init_aux: source = \n%s", source);

  hiprtcProgram program = NULL;
  hiprtcCreateProgram(&program, source, NULL, 0, NULL, NULL);
  hiprtcResult status = hiprtcCompileProgram(program, 0, NULL);
  if (status != HIPRTC_SUCCESS) {
    const char *error = hiprtcGetErrorString(status);

    size_t size;
    hiprtcGetProgramLogSize(program, &size);
    char *log = gemv_calloc(char, size + 1);
    hiprtcGetProgramLog(program, log);

    fprintf(stderr, "error = %s\n", error);
    fprintf(stderr, "log = %s\n", log);
    gemv_free(&log);

    gemv_log(GEMV_ERROR, "hip_init_aux: Kernel compilation failed !");
  }

  size_t size;
  hiprtcGetCodeSize(program, &size);
  char *binary_data = gemv_calloc(char, size + 1);
  hiprtcGetCode(program, binary_data);
  hiprtcDestroyProgram(&program);

  check_hip_runtime(hipModuleLoadData(&module, binary_data));
  gemv_free(&binary_data);
  check_hip_runtime(hipModuleGetFunction(&kernel, module, "gemv"));
}

static void hip_init(struct gemv_backend_t *backend,
                     const struct gemv_t *gemv) {
  gemv_log(GEMV_INFO, "hip_init: ...");
  if (initialized) return;

  backend->malloc = hip_malloc;
  backend->free = hip_free;
  backend->copy = hip_copy;
  backend->run = hip_run;
  backend->finalize = hip_finalize;

  hip_init_aux(gemv);

  initialized = 1;
  gemv_log(GEMV_INFO, "hip_init: done.");
}

void gemv_register_hip(void) { gemv_backend_register("hip", hip_init); }

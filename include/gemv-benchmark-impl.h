#if !defined(__GEMV_BENCHMARK_IMPL_H__)
#define __GEMV_BENCHMARK_IMPL_H__

#include "gemv-benchmark-defs.h"
#include "gemv-benchmark.h"

#include <assert.h>

// Dynamic memory allocation function.
#define gemv_benchmark_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
GEMV_BENCHMARK_INTERN void gemv_benchmark_free_(void **p);
#define gemv_benchmark_free(p) gemv_benchmark_free_((void **)p)

GEMV_BENCHMARK_INTERN void gemv_benchmark_log_init(int verbose);

GEMV_BENCHMARK_INTERN void gemv_benchmark_log(int level, const char *fmt, ...);

GEMV_BENCHMARK_INTERN void gemv_benchmark_error(const char *fmt, ...);

struct gemv_benchmark_t {
  int verbose;
  int device;
  int size;
  int num_repeats;
  char backend[32];
};

struct gemv_benchmark_backend_t {
  char name[32];
  void (*init)(int device, int n, const float *A, const float *x);
  void (*benchmark)(int num_repeats, float *y);
  void (*finalize)(void);
};

GEMV_BENCHMARK_INTERN void gemv_benchmark_register_backend(
    const char *name,
    void (*init)(int device, int n, const float *A, const float *x),
    void (*benchmark)(int num_repeats, float *y), void (*finalize)(void));

GEMV_BENCHMARK_INTERN void
gemv_benchmark_run_backend(const struct gemv_benchmark_t *benchmark);

GEMV_BENCHMARK_INTERN void gemv_benchmark_unregister_backends(void);

#define GEMV_BENCHMARK_BACKEND(name)                                           \
  GEMV_BENCHMARK_INTERN void gemv_benchmark_register_##name(void);

#include "gemv-benchmark-backend-list.h"

#undef GEMV_BENCHMARK_BACKEND

#endif // GEMV_BENCHMARK_IMPL_H

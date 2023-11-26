#include "gemv-benchmark.h"

#define GEMV_BENCHMARK_BACKEND(name)                                           \
  GEMV_BENCHMARK_INTERN void gemv_benchmark_register_##name(void)              \
      __attribute__((weak));                                                   \
  void gemv_benchmark_register_##name(void) { return; }

#include "gemv-benchmark-backend-list.h"

#undef GEMV_BENCHMARK_BACKEND

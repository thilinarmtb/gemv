#include "gemv-benchmark.h"

#include <assert.h>
#include <stddef.h>

int main(int argc, char *argv[]) {
  struct gemv_benchmark_t *handle = gemv_benchmark_init(&argc, &argv);
  assert((void *)handle != NULL);

  gemv_benchmark_run(handle);

  gemv_benchmark_finalize(&handle);
  assert((void *)handle == NULL);

  return 0;
}

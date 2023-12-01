#include "gemv-benchmark-impl.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void gemv_benchmark_free_(void **p) { free(*p), *p = NULL; }

static struct gemv_benchmark_backend_t *backend_list = NULL;
static int backend_count = 0;
static int backend_max_count = 0;

void gemv_benchmark_register_backend(
    const char *name,
    void (*init)(int device, int n, const float *A, const float *x),
    void (*benchmark)(int num_repeats, float *y), void (*finalize)(void)) {
  if (backend_count == backend_max_count) {
    backend_max_count += backend_max_count / 2 + 1;
    backend_list =
        realloc(backend_list, backend_max_count * sizeof(*backend_list));
  }

  strncpy(backend_list[backend_count].name, name, 32);
  backend_list[backend_count].init = init;
  backend_list[backend_count].benchmark = benchmark;
  backend_list[backend_count].finalize = finalize;
  backend_count++;
}

void gemv_benchmark_run_backend(const struct gemv_benchmark_t *benchmark) {
  int backend = -1;
  for (int i = 0; i < backend_count; i++) {
    if (strcmp(backend_list[i].name, benchmark->backend) == 0) {
      backend = i;
      break;
    }
  }
  assert(backend >= 0);
  gemv_benchmark_log(benchmark->verbose, "run_backend: %s", benchmark->backend);

  float *A = gemv_benchmark_calloc(float, benchmark->size * benchmark->size);
  for (int i = 0; i < benchmark->size * benchmark->size; i++)
    A[i] = (float)rand() / RAND_MAX;

  float *x = gemv_benchmark_calloc(float, benchmark->size);
  for (int i = 0; i < benchmark->size; i++)
    x[i] = (float)rand() / RAND_MAX;

  float *y = gemv_benchmark_calloc(float, benchmark->size);

  // Initialize the backend:
  backend_list[backend].init(benchmark->device, benchmark->size, A, x);

  // Run the benchmark:
  clock_t start = clock();
  backend_list[backend].benchmark(benchmark->num_repeats, y);
  clock_t end = clock();
  double elapsed =
      (double)(end - start) / (CLOCKS_PER_SEC * benchmark->num_repeats);

  // Check correctness:
  float *y_ref = gemv_benchmark_calloc(float, benchmark->size);
  for (int i = 0; i < benchmark->size; i++) {
    float sum = 0.0f;
    for (int j = 0; j < benchmark->size; j++)
      sum += A[i * benchmark->size + j] * x[j];
    y_ref[i] = sum;
  }
  for (int i = 0; i < benchmark->size; i++) {
    if (fabs(y[i] - y_ref[i]) / y_ref[i] > 1e-5)
      gemv_benchmark_error("run_backend: %s: y[%d] = %f != %f",
                           benchmark->backend, i, y[i], y_ref[i]);
  }
  gemv_benchmark_free(&y_ref);

  gemv_benchmark_log(benchmark->verbose, "run_backend: elapsed: %f", elapsed);

  backend_list[backend].finalize();

  gemv_benchmark_free(&A);
  gemv_benchmark_free(&x);
  gemv_benchmark_free(&y);
}

void gemv_benchmark_unregister_backends(void) {
  for (int i = 0; i < backend_count; i++) {
    if (backend_list[i].finalize)
      backend_list[i].finalize();
  }
  backend_count = 0;
  backend_max_count = 0;
  gemv_benchmark_free(&backend_list);
}

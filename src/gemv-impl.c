#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemv-impl.h"

void gemv_free_(void **p) { free(*p), *p = NULL; }

static struct gemv_backend_t *backend_list = NULL;
static int backend_count = 0;
static int backend_max_count = 0;

void gemv_register_backend(const char *name,
                           void (*init)(int device, int n, const float *A),
                           void (*gemv)(float *y, const float *x),
                           void (*finalize)(void)) {
  if (backend_count == backend_max_count) {
    backend_max_count += backend_max_count / 2 + 1;
    backend_list =
        realloc(backend_list, backend_max_count * sizeof(*backend_list));
  }

  strncpy(backend_list[backend_count].name, name, 32);
  backend_list[backend_count].init = init;
  backend_list[backend_count].gemv = gemv;
  backend_list[backend_count].finalize = finalize;
  backend_count++;
}

void gemv_run_backend(const struct gemv_t *gemv) {
  int backend = -1;
  for (int i = 0; i < backend_count; i++) {
    if (strcmp(backend_list[i].name, gemv->backend) == 0) {
      backend = i;
      break;
    }
  }
  assert(backend >= 0);
  gemv_log(gemv->verbose, "run_backend: %s", gemv->backend);

  float *A = gemv_calloc(float, gemv->size * gemv->size);
  for (int i = 0; i < gemv->size * gemv->size; i++)
    A[i] = (float)rand() / RAND_MAX;

  float *x = gemv_calloc(float, gemv->size);
  for (int i = 0; i < gemv->size; i++)
    x[i] = (float)rand() / RAND_MAX;

  float *y = gemv_calloc(float, gemv->size);

  // Initialize the backend:
  backend_list[backend].init(gemv->device, gemv->size, A);

  // Run the gemv:
  clock_t start = clock();
  backend_list[backend].gemv(y, x);
  clock_t end = clock();
  double elapsed = (double)(end - start) / (CLOCKS_PER_SEC * gemv->num_repeats);

  // Check correctness:
  float *y_ref = gemv_calloc(float, gemv->size);
  for (int i = 0; i < gemv->size; i++) {
    float sum = 0.0f;
    for (int j = 0; j < gemv->size; j++)
      sum += A[i * gemv->size + j] * x[j];
    y_ref[i] = sum;
  }
  for (int i = 0; i < gemv->size; i++) {
    if (fabs(y[i] - y_ref[i]) / y_ref[i] > 1e-5)
      gemv_error("run_backend: %s: y[%d] = %f != %f", gemv->backend, i, y[i],
                 y_ref[i]);
  }
  gemv_free(&y_ref);

  gemv_log(gemv->verbose, "run_backend: elapsed: %f", elapsed);

  backend_list[backend].finalize();

  gemv_free(&A), gemv_free(&x), gemv_free(&y);
}

void gemv_unregister_backends(void) {
  for (int i = 0; i < backend_count; i++) {
    if (backend_list[i].finalize)
      backend_list[i].finalize();
  }

  backend_count = backend_max_count = 0, gemv_free(&backend_list);
}

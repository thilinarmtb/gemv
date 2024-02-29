#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemv-impl.h"

void gemv_free_(void **p) { free(*p), *p = NULL; }

static struct gemv_backend_t *backend_list = NULL;
static unsigned backend_count = 0, backend_max_count = 0;
static int backend_active = -1;

void gemv_backend_register(const char *name,
                           void (*init)(int device, int n, const float *A),
                           void (*copy)(void *dest, const void *src,
                                        size_t count,
                                        gemv_direction_t direction),
                           void (*run)(float *y, const float *x),
                           void (*finalize)(void)) {
  if (backend_count == backend_max_count) {
    backend_max_count += backend_max_count / 2 + 1;
    backend_list =
        realloc(backend_list, backend_max_count * sizeof(*backend_list));
  }

  strncpy(backend_list[backend_count].name, name, GEMV_MAX_BACKEND_LENGTH);
  backend_list[backend_count].init = init;
  backend_list[backend_count].copy = copy;
  backend_list[backend_count].run = run;
  backend_list[backend_count].finalize = finalize;
  backend_count++;
}

void gemv_set_backend_impl(struct gemv_t *gemv, const char *backend) {
  size_t backend_length = strnlen(backend, GEMV_MAX_BACKEND_LENGTH);
  char backend_lower[GEMV_MAX_BACKEND_LENGTH + 1];
  for (unsigned i = 0; i < backend_length; i++)
    backend_lower[i] = tolower(backend[i]);
  backend_lower[backend_length] = '\0';

  gemv->backend = -1;
  for (unsigned i = 0; i < backend_count; i++) {
    if (strncmp(backend_lower, backend_list[i].name, GEMV_MAX_BACKEND_LENGTH) ==
        0) {
      gemv->backend = i;
      break;
    }
  }
}

void gemv_backend_init(int backend, int device, size_t size, const double *A) {}

void gemv_backend_run(float *y, const float *x) {
  gemv_assert(backend_active >= 0,
              "gemv_backend_init: A backend is not initialized.");
}

void gemv_backend_finalize(void) {}

void gemv_check_impl(const struct gemv_t *gemv) {
  assert(gemv->backend >= 0);
  assert(gemv->device >= 0);
  gemv_log(gemv->verbose, "gemv_check: %s", gemv->backend);

  // Initialize the matrix and RHS.
  srand(time(NULL));

  const size_t size = 8192;
  double *A = gemv_calloc(double, size *size);
  for (unsigned i = 0; i < size * size; i++) A[i] = (double)rand() / RAND_MAX;

  float *x = gemv_calloc(float, size);
  for (unsigned i = 0; i < size; i++) x[i] = (float)rand() / RAND_MAX;

  float *y = gemv_calloc(float, size);

  // Initialize the backend:
  gemv_backend_init(gemv->backend, gemv->device, size, A);

  // Run the gemv:
  gemv_backend_run(y, x);

  // Check correctness:
  float *y_ref = gemv_calloc(float, size);
  for (unsigned i = 0; i < size; i++) {
    float sum = 0.0f;
    for (unsigned j = 0; j < size; j++) sum += A[i * size + j] * x[j];
    y_ref[i] = sum;
  }
  for (unsigned i = 0; i < size; i++) {
    if (fabs(y[i] - y_ref[i]) / y_ref[i] > 1e-5)
      gemv_log(GEMV_ERROR, "gemv_check: y[%d] = %f != %f", i, y[i], y_ref[i]);
  }
  gemv_free(&y_ref);

  gemv_log(gemv->verbose, "gemv_check: pass.");

  gemv_backend_finalize();

  gemv_free(&A), gemv_free(&x), gemv_free(&y);
}

void gemv_backend_deregister(void) {
  for (unsigned i = 0; i < backend_count; i++)
    if (backend_list[i].finalize) backend_list[i].finalize();

  backend_count = backend_max_count = 0, gemv_free(&backend_list);
}

#if !defined(__GEMV_IMPL_H__)
#define __GEMV_IMPL_H__

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "gemv-defs.h"
#include "gemv.h"

// Dynamic memory allocation function.
#define gemv_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
GEMV_INTERN void gemv_free_(void **p);
#define gemv_free(p) gemv_free_((void **)p)

GEMV_INTERN void gemv_log_init(int verbose);

GEMV_INTERN void gemv_log(int level, const char *fmt, ...);

GEMV_INTERN void gemv_error(const char *fmt, ...);

struct gemv_t {
  int verbose;
  int device;
  int size;
  int num_repeats;
  char backend[32];
};

struct gemv_backend_t {
  char name[32];
  void (*init)(int device, int n, const float *A, const float *x);
  void (*benchmark)(int num_repeats, float *y);
  void (*finalize)(void);
};

GEMV_INTERN void gemv_register_backend(
    const char *name,
    void (*init)(int device, int n, const float *A, const float *x),
    void (*benchmark)(int num_repeats, float *y), void (*finalize)(void));

GEMV_INTERN void gemv_run_backend(const struct gemv_t *benchmark);

GEMV_INTERN void gemv_unregister_backends(void);

#define GEMV_BACKEND(name) GEMV_INTERN void gemv_register_##name(void);

#include "gemv-backend-list.h"

#undef GEMV_BACKEND

#endif // GEMV_IMPL_H

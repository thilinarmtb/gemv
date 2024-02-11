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

GEMV_INTERN void gemv_log(const gemv_verbose_t verbose, const char *fmt, ...);

GEMV_INTERN void gemv_assert(int cond, const char *fmt, ...);

struct gemv_t {
  int verbose, device, backend;
  gemv_precision_t precision;
};

struct gemv_backend_t {
  char name[32];
  void (*init)(int, int, const float *);
  void (*copy)(void *, const void *, size_t, gemv_direction_t);
  void (*gemv)(float *, const float *);
  void (*finalize)(void);
};

GEMV_INTERN void gemv_register_backend(
    const char *name, void (*init)(int device, int n, const float *A),
    void (*copy)(void *dest, const void *src, size_t count,
                 gemv_direction_t direction),
    void (*gemv)(float *y, const float *x), void (*finalize)(void));

GEMV_INTERN void gemv_backend_init(int backend, int device, size_t size,
                                   const double *A);

GEMV_INTERN void gemv_backend_run(float *y, const float *x);

GEMV_INTERN void gemv_backend_finalize(void);

GEMV_INTERN void gemv_deregister_backends(void);

#define GEMV_BACKEND(name) GEMV_INTERN void gemv_register_##name(void);
#include "gemv-backend-list.h"
#undef GEMV_BACKEND

#endif // __GEMV_IMPL_H__

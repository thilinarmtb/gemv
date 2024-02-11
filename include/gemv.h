#if !defined(__GEMV_H__)
#define __GEMV_H__

#define GEMV_VISIBILITY(mode) __attribute__((visibility(#mode)))

#if defined(__cplusplus)
#define GEMV_EXTERN extern "C" GEMV_VISIBILITY(default)
#else
#define GEMV_EXTERN extern GEMV_VISIBILITY(default)
#endif

#if defined(__cplusplus)
#define GEMV_INTERN extern "C" GEMV_VISIBILITY(hidden)
#else
#define GEMV_INTERN extern GEMV_VISIBILITY(hidden)
#endif

#include <stddef.h>

GEMV_EXTERN struct gemv_t *gemv_init(int *argc, char ***argv);

GEMV_EXTERN void gemv_set_verbose(int verbose);

GEMV_EXTERN void gemv_set_device(struct gemv_t *gemv, int device);

GEMV_EXTERN void gemv_set_backend(struct gemv_t *gemv, const char *backend);

GEMV_EXTERN void gemv_set_matrix(struct gemv_t *gemv, float *A);

typedef enum { GEMV_H2D = 0, GEMV_D2H } gemv_direction_t;

GEMV_EXTERN void gemv_copy(void *dst, const void *src, size_t count,
                           const gemv_direction_t direction);

GEMV_EXTERN void gemv(float *y, const struct gemv_t *gemv, const float *x);

GEMV_EXTERN void gemv_check(const struct gemv_t *gemv);

GEMV_EXTERN void gemv_finalize(struct gemv_t **gemv);

#endif // __GEMV_H__

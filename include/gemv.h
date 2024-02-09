#if !defined(__GEMV__)
#define __GEMV__

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

GEMV_EXTERN struct gemv_t *gemv_init(int *argc, char ***argv);

GEMV_EXTERN void gemv_run(const struct gemv_t *gemv);

GEMV_EXTERN void gemv_finalize(struct gemv_t **gemv);

#endif // GEMV

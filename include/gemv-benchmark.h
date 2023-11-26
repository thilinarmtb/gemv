#if !defined(__GEMV_BENCHMARK__)
#define __GEMV_BENCHMARK__

#define GEMV_BENCHMARK_VISIBILITY(mode) __attribute__((visibility(#mode)))

#if defined(__cplusplus)
#define GEMV_BENCHMARK_EXTERN extern "C" GEMV_BENCHMARK_VISIBILITY(default)
#else
#define GEMV_BENCHMARK_EXTERN extern GEMV_BENCHMARK_VISIBILITY(default)
#endif

#if defined(__cplusplus)
#define GEMV_BENCHMARK_INTERN extern "C" GEMV_BENCHMARK_VISIBILITY(hidden)
#else
#define GEMV_BENCHMARK_INTERN extern GEMV_BENCHMARK_VISIBILITY(hidden)
#endif

struct gemv_bencmark_t;

GEMV_BENCHMARK_EXTERN struct gemv_benchmark_t *
gemv_benchmark_init(int *argc, char ***argv);

GEMV_BENCHMARK_EXTERN void
gemv_benchmark_run(const struct gemv_benchmark_t *benchmark);

GEMV_BENCHMARK_EXTERN void
gemv_benchmark_finalize(struct gemv_benchmark_t **benchmark);

#endif // GEMV_BENCHMARK

#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemv-impl.h"

static void print_help(const char *name, int status) {
  FILE *fp = (status == EXIT_SUCCESS) ? stdout : stderr;
  fprintf(fp, "Usage: %s [OPTIONS]\n", name);
  fprintf(fp, "Options:\n");
  fprintf(fp,
          "  --gemv-verbose=<verbose level>, Verbose level (0, 1, 2, ...).\n");
  fprintf(fp, "  --gemv-device=<device id>, Device ID (0, 1, 2, ...).\n");
  fprintf(fp,
          "  --gemv-backend=<backend>, Backend (CUDA, HIP, OpenCL, etc.).\n");
  fprintf(fp, "  --gemv-help, Prints this help message and exit.\n");
  fflush(fp);
  exit(status);
}

static void parse_opts(struct gemv_t *gemv, int *argc, char ***argv_) {
  static struct option long_options[] = {
      {"gemv-verbose", optional_argument, 0, 10},
      {"gemv-device", optional_argument, 0, 20},
      {"gemv-backend", required_argument, 0, 30},
      {"gemv-precision", required_argument, 0, 40},
      {"gemv-help", no_argument, 0, 99},
      {0, 0, 0, 0}};

  // Default values for optional arguments.
  int device = GEMV_DEFAULT_DEVICE;
  gemv_verbose_t verbose = GEMV_DEFAULT_VERBOSE;
  gemv_precision_t precision = GEMV_DEFAULT_PRECISION;
  char backend[GEMV_MAX_BACKEND_LENGTH + 1];
  strncpy(backend, GEMV_DEFAULT_BACKEND, GEMV_MAX_BACKEND_LENGTH);

  if (argc == NULL || *argc == 0 || argv_ == NULL) goto set_options;

  char **argv = *argv_;
  for (;;) {
    int c = getopt_long(*argc, argv, "", long_options, NULL);
    if (c == -1) break;

    switch (c) {
    case 10: verbose = atoi(optarg); break;
    case 20: device = atoi(optarg); break;
    case 30: strncpy(backend, optarg, GEMV_MAX_BACKEND_LENGTH); break;
    case 40: precision = atoi(optarg); break;
    case 99: print_help(argv[0], EXIT_SUCCESS); break;
    default: print_help(argv[0], EXIT_FAILURE); break;
    }
  }

  for (int i = optind; i < *argc; i++) argv[i - optind] = argv[i];
  *argc -= optind;

set_options:
  gemv_set_verbose(verbose);
  gemv_set_device(gemv, device);
  gemv_set_backend(gemv, backend);
  gemv_set_precision(gemv, precision);
}

struct gemv_t *gemv_init(int *argc, char ***argv) {
  // Register all the backends.
#define GEMV_BACKEND(name) gemv_register_##name();
#include "backends/gemv-backend-list.h"
#undef GEMV_BACKEND

  // Initialize the gemv_t struct.
  struct gemv_t *gemv = gemv_calloc(struct gemv_t, 1);

  // Parse command line options if present.
  parse_opts(gemv, argc, (char ***)argv);

  // Log info if verbose level is set.
  gemv_log(GEMV_INFO, "gemv_init: device: %d", gemv->device);
  gemv_log(GEMV_INFO, "gemv_init: backend: %d", gemv->backend);
  gemv_log(GEMV_INFO, "gemv_init: precision: %d", gemv->precision);

  return gemv;
}

void gemv_set_verbose(const gemv_verbose_t verbose) {
  gemv_set_verbose_impl(verbose);
}

void gemv_set_device(struct gemv_t *gemv, int device) { gemv->device = device; }

void gemv_set_backend(struct gemv_t *gemv, const char *backend) {
  gemv_set_backend_impl(gemv, backend);
}

void gemv_set_matrix(struct gemv_t *gemv, const unsigned n, const unsigned m,
                     const double *A) {
  gemv->m = m, gemv->n = n;
  gemv->A = gemv_realloc(gemv->A, double, m *n);
  memcpy(gemv->A, A, sizeof(double) * m * n);
}

void gemv_set_precision(struct gemv_t *gemv, const gemv_precision_t precision) {
  gemv->precision = precision;
}

void gemv_check(const struct gemv_t *gemv) { gemv_check_impl(gemv); }

void gemv_copy(void *dst, const void *src, size_t count,
               const gemv_direction_t direction) {}

void gemv_run(void *y, const struct gemv_t *gemv, const void *x) {}

void gemv_finalize(struct gemv_t **gemv) {
  gemv_backend_deregister();
  gemv_free(&(*gemv)->A);
  gemv_free(gemv);
}

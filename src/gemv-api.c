#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemv-impl.h"

static void print_help(const char *name, int status) {
  FILE *fp = (status == EXIT_SUCCESS) ? stdout : stderr;
  fprintf(fp, "Usage: %s [OPTIONS]\n", name);
  fprintf(fp, "Options:\n");
  fprintf(fp, "  --verbose=<verbose level>, Verbose level (0, 1, 2, ...).\n");
  fprintf(fp, "  --device=<device id>, Device ID (0, 1, 2, ...).\n");
  fprintf(fp, "  --backend=<backend>, Backend (CUDA, HIP, OpenCL, etc.).\n");
  fprintf(fp, "  --help, Prints this help message and exit.\n");
  fflush(fp);
  exit(status);
}

inline static void set_backend(struct gemv_t *gemv, const char *backend) {
  size_t len = strnlen(backend, 32);
  for (uint i = 0; i < len; i++)
    gemv->backend[i] = tolower(backend[i]);
}

static void parse_opts(struct gemv_t *gemv, int *argc, char ***argv_) {
  static struct option long_options[] = {{"verbose", optional_argument, 0, 10},
                                         {"device", optional_argument, 0, 20},
                                         {"backend", required_argument, 0, 30},
                                         {"help", no_argument, 0, 99},
                                         {0, 0, 0, 0}};

  // Default values for optional arguments.
  gemv->verbose = GEMV_DEFAULT_VERBOSE;
  gemv->device = GEMV_DEFAULT_DEVICE;
  strncpy(gemv->backend, GEMV_DEFAULT_BACKEND, 16);

  if (argc == NULL || *argc == 0 || argv_ == NULL) return;

  char **argv = *argv_;
  for (;;) {
    int c = getopt_long(*argc, argv, "", long_options, NULL);
    if (c == -1) break;

    switch (c) {
    case 10:
      gemv->verbose = atoi(optarg);
      break;
    case 20:
      gemv->device = atoi(optarg);
      break;
    case 30:
      set_backend(gemv, optarg);
      break;
    case 99:
      print_help(argv[0], EXIT_SUCCESS);
      break;
    default:
      print_help(argv[0], EXIT_FAILURE);
      break;
    }
  }

  // Remove parsed arguments from argv. We just need to update the pointers
  // since command line arguments are not transient and available until the
  // end of the program.
  for (int i = optind; i < *argc; i++)
    argv[i - optind] = argv[i];
  *argc -= optind;
}

struct gemv_t *gemv_init(int *argc, char ***argv) {
#define GEMV_BACKEND(name) gemv_register_##name();
#include "gemv-backend-list.h"
#undef GEMV_BACKEND

  // Initialize the random number generator.
  srand(time(NULL));

  // Initialize the gemv_t struct.
  struct gemv_t *gemv = gemv_calloc(struct gemv_t, 1);
  parse_opts(gemv, argc, (char ***)argv);

  gemv_set_verbose(gemv->verbose);

  gemv_log(gemv->verbose, "parse_opts: verbose: %d", gemv->verbose);
  gemv_log(gemv->verbose, "parse_opts: device: %d", gemv->device);
  gemv_log(gemv->verbose, "parse_opts: backend: %s", gemv->backend);

  return gemv;
}

void gemv_set_device(int device) {}

void gemv_set_backend(const char *backend) {}

void gemv_check(const struct gemv_t *gemv) { gemv_check_backend(gemv); }

void gemv(float *y, const struct gemv_t *gemv, const float *x);

void gemv_copy(void *dst, const void *src, size_t count,
               const gemv_direction_t direction) {}

void gemv_finalize(struct gemv_t **gemv) {
  gemv_unregister_backends();
  gemv_free(gemv);
}

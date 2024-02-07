#include "gemv-benchmark-impl.h"

#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_help(const char *name, int status) {
  FILE *fp = (status == EXIT_SUCCESS) ? stdout : stderr;
  fprintf(fp, "Usage: %s [OPTIONS]\n", name);
  fprintf(fp, "Options:\n");
  fprintf(fp, "  --verbose=<verbose level>, Verbose level (0, 1, 2, ...).\n");
  fprintf(fp, "  --device=<device id>, Device ID (0, 1, 2, ...).\n");
  fprintf(fp, "  --num-repeats=<iters>, Number of repeats (1, 2, 3, ...)\n");
  fprintf(fp, "  --backend=<backend>, Backend (CUDA, OpenCL, nomp, etc.).\n");
  fprintf(fp, "  --size, Number of elements (1, 2, 3, ...).\n");
  fprintf(fp, "  --help, Prints this help message and exit.\n");
  fflush(fp);
  exit(status);
}

inline static void set_backend(struct gemv_benchmark_t *gemv_benchmark,
                               const char *backend) {
  size_t len = strnlen(backend, 32);
  for (uint i = 0; i < len; i++)
    gemv_benchmark->backend[i] = tolower(backend[i]);
}

static void parse_opts(struct gemv_benchmark_t *gemv_benchmark, int *argc,
                       char ***argv_) {
  static struct option long_options[] = {
      {"verbose", optional_argument, 0, 10},
      {"device", optional_argument, 0, 20},
      {"num-repeats", optional_argument, 0, 30},
      {"backend", required_argument, 0, 40},
      {"size", required_argument, 0, 50},
      {"help", no_argument, 0, 99},
      {0, 0, 0, 0}};

  // Default values for optional arguments.
  gemv_benchmark->verbose = GEMV_VERBOSE;
  gemv_benchmark->device = GEMV_DEVICE;
  gemv_benchmark->num_repeats = GEMV_NUM_REPEATS;

  // Set invalid values for required arguments so we can check if they were
  // initialized later.
  gemv_benchmark->size = -1;
  strncpy(gemv_benchmark->backend, "", 1);

  char **argv = *argv_;
  for (;;) {
    int c = getopt_long(*argc, argv, "", long_options, NULL);
    if (c == -1)
      break;

    switch (c) {
    case 10:
      gemv_benchmark->verbose = atoi(optarg);
      break;
    case 20:
      gemv_benchmark->device = atoi(optarg);
      break;
    case 30:
      gemv_benchmark->num_repeats = atoi(optarg);
      break;
    case 40:
      set_backend(gemv_benchmark, optarg);
      break;
    case 50:
      gemv_benchmark->size = atoi(optarg);
      break;
    case 99:
      print_help(argv[0], EXIT_SUCCESS);
      break;
    default:
      print_help(argv[0], EXIT_FAILURE);
      break;
    }
  }

  if (gemv_benchmark->size <= 0)
    gemv_benchmark_error("parse_opts: size is not set !");
  if (gemv_benchmark->backend[0] == '\0')
    gemv_benchmark_error("parse_opts: backend is not set !");

  // Remove parsed arguments from argv. We just need to update the pointers
  // since command line arguments are not transient and available until the
  // end of the program.
  for (int i = optind; i < *argc; i++)
    argv[i - optind] = argv[i];
  *argc -= optind;

  gemv_benchmark_log(gemv_benchmark->verbose, "parse_opts: verbose: %d",
                     gemv_benchmark->verbose);
  gemv_benchmark_log(gemv_benchmark->verbose, "parse_opts: device: %d",
                     gemv_benchmark->device);
  gemv_benchmark_log(gemv_benchmark->verbose, "parse_opts: num_repeats: %d",
                     gemv_benchmark->num_repeats);
  gemv_benchmark_log(gemv_benchmark->verbose, "parse_opts: size: %d",
                     gemv_benchmark->size);
  gemv_benchmark_log(gemv_benchmark->verbose, "parse_opts: backend: %s",
                     gemv_benchmark->backend);
}

struct gemv_benchmark_t *gemv_benchmark_init(int *argc, char ***argv) {
  gemv_benchmark_log_init(1);

#define GEMV_BENCHMARK_BACKEND(name) gemv_benchmark_register_##name();
#include "gemv-benchmark-backend-list.h"
#undef GEMV_BENCHMARK_BACKEND

  // Initialize the random number generator.
  srand(time(NULL));

  // Initialize the gemv_benchmark_t struct.
  struct gemv_benchmark_t *gemv_benchmark =
      gemv_benchmark_calloc(struct gemv_benchmark_t, 1);
  parse_opts(gemv_benchmark, argc, (char ***)argv);
  return gemv_benchmark;
}

void gemv_benchmark_run(const struct gemv_benchmark_t *gemv_benchmark) {
  gemv_benchmark_run_backend(gemv_benchmark);
}

void gemv_benchmark_finalize(struct gemv_benchmark_t **gemv_benchmark) {
  gemv_benchmark_unregister_backends();
  gemv_benchmark_free(gemv_benchmark);
}

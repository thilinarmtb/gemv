#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "gemv-impl.h"

static int log_level = 0;

void gemv_set_verbose(int level) { log_level = level; }

void gemv_log(int level, const char *fmt, ...) {
  if (level >= log_level) {
    va_list args;
    va_start(args, fmt);
    char buf[BUFSIZ];
    vsnprintf(buf, BUFSIZ, fmt, args);
    va_end(args);

    fprintf(stderr, "%s\n", buf);
    fflush(stderr);
  }
}

void gemv_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char buf[BUFSIZ];
  vsnprintf(buf, BUFSIZ, fmt, args);
  va_end(args);

  fprintf(stderr, "%s\n", buf);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

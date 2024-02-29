#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "gemv-impl.h"

static gemv_verbose_t log_level = 0;

void gemv_set_verbose_impl(const gemv_verbose_t level) {
  if (level < GEMV_MUTE || level > GEMV_ERROR) {
    fprintf(stderr, "gemv_set_verbose: Invalid verbose level: %d\n", level);
    exit(EXIT_FAILURE);
  }

  log_level = level;
}

void gemv_log(const gemv_verbose_t level, const char *fmt, ...) {
  if (level < log_level) return;

  va_list args;
  va_start(args, fmt);
  char buf[BUFSIZ];
  vsnprintf(buf, BUFSIZ, fmt, args);
  va_end(args);

  fprintf(stderr, "%s\n", buf);
  fflush(stderr);

  if (level == GEMV_ERROR) exit(EXIT_FAILURE);
}

void gemv_assert(int cond, const char *fmt, ...) {}

#include <assert.h>
#include <stddef.h>

#include "gemv.h"

int main(int argc, char *argv[]) {
  struct gemv_t *handle = gemv_init(&argc, &argv);

  gemv_set_verbose(GEMV_INFO);
  gemv_set_precision(handle, GEMV_FP64);

  const double A[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  gemv_set_matrix(handle, 3, 3, A);

  gemv_init_session(handle);

  gemv_finalize_session();

  gemv_finalize(&handle);

  return 0;
}

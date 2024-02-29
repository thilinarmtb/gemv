#include <assert.h>
#include <stddef.h>

#include "gemv.h"

int main(int argc, char *argv[]) {
  struct gemv_t *handle = gemv_init(&argc, &argv);
  assert((void *)handle != NULL);

  gemv_check(handle);

  gemv_finalize(&handle);
  assert((void *)handle == NULL);

  return 0;
}

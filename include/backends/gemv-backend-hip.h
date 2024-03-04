#if !defined(__GEMV_BACKEND_HIP_H__)
#define __GEMV_BACKEND_HIP_H__

#include "gemv-backend.h"

#include <hip/hip_runtime.h>

static inline void check_hip_runtime_(hipError_t err, const char *file,
                                      const unsigned line) {
  if (err == hipSuccess) return;
  gemv_log(GEMV_ERROR, "HIP runtime error:\"%s\" in file: %s line: %u",
           hipGetErrorString(err), file, line);
}

#define check_hip_runtime(call) check_hip_runtime_((call), __FILE__, __LINE__)

static void hip_copy(void *dest, const void *src, size_t count,
                     gemv_direction_t direction) {
  enum hipMemcpyKind kind = hipMemcpyDefault;
  switch (direction) {
  case GEMV_D2H: kind = hipMemcpyDeviceToHost; break;
  case GEMV_H2D: kind = hipMemcpyHostToDevice; break;
  default:
    gemv_log(GEMV_ERROR, "hip_copy: Invalid direction = %d", direction);
    break;
  }

  check_hip_runtime(hipMemcpy(dest, src, count, kind));
}

#endif // __GEMV_BACKEND_HIP_H__

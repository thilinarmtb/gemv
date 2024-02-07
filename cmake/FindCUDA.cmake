find_package(CUDAToolkit)

if (TARGET CUDA::toolkit)
  add_library(gemv::CUDA INTERFACE IMPORTED)
  target_link_libraries(gemv::CUDA INTERFACE CUDA::cudart CUDA::cuda_driver
    CUDA::cublas)
endif()

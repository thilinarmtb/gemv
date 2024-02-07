cmake_minimum_required(VERSION 3.22)
project(gemv VERSION 0.0.1 DESCRIPTION "Platform independent gemv on GPUs"
  LANGUAGES C)

# Build Options:
option(ENABLE_CUDA "Build CUDA backend" OFF)
option(ENABLE_HIP "Build HIP backend" OFF)

# Build defaults:
set(GEMV_VERBOSE 1)
set(GEMV_DEVICE 0)
set(GEMV_NUM_REPEATS 100)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

configure_file(${CMAKE_SOURCE_DIR}/include/gemv-benchmark-defs.h.in
  ${CMAKE_SOURCE_DIR}/include/gemv-benchmark-defs.h @ONLY)

set(GEMV_SOURCES src/gemv-benchmark-api.c src/gemv-benchmark-impl.c
  src/gemv-benchmark-hip.c src/gemv-benchmark-dummy.c src/gemv-benchmark-log.c)

find_package(HIP)
if (NOT DEFINED ROCM_PATH)
  if (NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm path")
  else()
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm path")
  endif()
endif()
set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake" ${CMAKE_MODULE_PATH})
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

add_library(gemv SHARED ${GEMV_SOURCES})
set_target_properties(gemv PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER include/gemv-benchmark.h)
target_include_directories(gemv PRIVATE include)
target_compile_options(gemv PRIVATE $<$<C_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wno-unused-function
  -Wno-unused-parameter -Wno-stringop-overflow>)

if (HIP_FOUND)
  target_link_libraries(gemv PRIVATE gemv::HIP)
endif()

macro(gemv_add_binary file_name)
  get_filename_component(target ${file_name} NAME_WE)
  add_executable(${target} ${file_name})
  set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION})
  target_compile_options(${target} PRIVATE $<$<C_COMPILER_ID:MSVC>:/W4 /MX>
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wno-unused-function
    -Wno-unused-parameter -Wno-stringop-overflow>)
  target_include_directories(${target} PRIVATE include)
  target_link_libraries(${target} PRIVATE gemv)
  install(TARGETS ${target} DESTINATION bin)
endmacro()

gemv_add_binary(bin/gemv-benchmark.c)

install(TARGETS gemv LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

# Add clang-format as a custom target if available.
find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-format ...")
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT} --dry-run -Werror -i ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-format check ...")
endif()

# Configure .clang-tidy.in to generate .clang-tidy.
configure_file(${CMAKE_SOURCE_DIR}/.clang-tidy.in ${CMAKE_SOURCE_DIR}/.clang-tidy @ONLY)
# Add clang-tidy as a custom target if available.
find_program(CLANG_TIDY NAMES clang-tidy)
if (CLANG_TIDY)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-tidy ...")
endif()

#!/bin/bash

function print_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --help Print this help message and exit."
  echo "  --cc <compiler> Set the compiler to use for the build."
  echo "  --build-type <Release|Debug> Build type."
  echo "  --build-dir <build directory> Build directory."
  echo "  --install-prefix <install prefix> Install prefix."
  echo "  --install Install the project."
  echo "  --format Format the source code with clang-format."
  echo "  --format-check Check if source formatting is compliant with clang-format."
  echo "  --tidy Run clang-tidy."
}

# Set default values.
: ${GEMV_BENCHMARK_CC:=cc}
: ${GEMV_BENCHMARK_BUILD_TYPE:=Release}
: ${GEMV_BENCHMARK_INSTALL_PREFIX:=`pwd`/install}
: ${GEMV_BENCHMARK_BUILD_DIR:=`pwd`/build}
: ${GEMV_BENCHMARK_INSTALL:=NO}
: ${GEMV_BENCHMARK_FORMAT:=NO}
: ${GEMV_BENCHMARK_FORMAT_CHECK:=NO}
: ${GEMV_BENCHMARK_TIDY:=NO}

# Handle command line arguments.
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      print_help
      exit 0
      ;;
    --cc)
      GEMV_BENCHMARK_CC="$2"
      shift
      shift
      ;;
    --build-type)
      GEMV_BENCHMARK_BUILD_TYPE="$2"
      shift
      shift
      ;;
    --build-dir)
      GEMV_BENCHMARK_BUILD_DIR="$2"
      shift
      shift
      ;;
    --install-prefix)
      GEMV_BENCHMARK_INSTALL_PREFIX="$2"
      shift
      shift
      ;;
    --install)
      GEMV_BENCHMARK_INSTALL="YES"
      shift
      ;;
    --format)
      GEMV_BENCHMARK_FORMAT="YES"
      shift
      ;;
    --format-check)
      GEMV_BENCHMARK_FORMAT_CHECK="YES"
      shift
      ;;
    --tidy)
      GEMV_BENCHMARK_TIDY="YES"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done
  
mkdir -p ${GEMV_BENCHMARK_BUILD_DIR} 2> /dev/null

cmake -DCMAKE_C_COMPILER=${GEMV_BENCHMARK_CC} \
  -DCMAKE_BUILD_TYPE=${GEMV_BENCHMARK_BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${GEMV_BENCHMARK_INSTALL_PREFIX} \
  -B ${GEMV_BENCHMARK_BUILD_DIR} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -S .
  
if [[ "${GEMV_BENCHMARK_FORMAT}" == "YES" ]]; then
  cmake --build ${GEMV_BENCHMARK_BUILD_DIR} --target format -j4
fi

if [[ "${GEMV_BENCHMARK_FORMAT_CHECK}" == "YES" ]]; then
  cmake --build ${GEMV_BENCHMARK_BUILD_DIR} --target format-check -j4
  if [[ $? -ne 0 ]]; then
    echo "Error: clang-format check failed."
    exit 1
  fi
fi

if [[ "${GEMV_BENCHMARK_TIDY}" == "YES" ]]; then
  cmake --build ${GEMV_BENCHMARK_BUILD_DIR} --target tidy -j4
  if [[ $? -ne 0 ]]; then
    echo "Error: clang-tidy failed."
    exit 1
  fi
fi

if [[ "${GEMV_BENCHMARK_INSTALL}" == "YES" ]]; then
  cmake --build ${GEMV_BENCHMARK_BUILD_DIR} --target install -j4
  if [[ $? -ne 0 ]]; then
    echo "Error: Installing failed."
    exit 1
  fi
fi

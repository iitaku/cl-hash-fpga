# Global settings
project(cl-hash)
cmake_minimum_required(VERSION 2.8)

set(BUILD_ALTERA false CACHE BOOL "build altera binary")

# OpenCL must be required
include(FindOpenCL)
if(NOT ${OpenCL_FOUND})
  message(FATAL_ERROR "Cannot found OpenCL environment")
endif()

# Set path to header
include_directories(OpenCL_INCLUDE_DIRS)

# Generate executable
add_executable(cl-hash cl-hash.cc)
target_link_libraries(cl-hash ${OpenCL_LIBRARIES})

# Generate fpga executable
if(BUILD_ALTERA)
  execute_process(
    COMMAND aocl compile-config
    OUTPUT_VARIABLE AOCL_COMPILE_CONFIG
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  STRING(REPLACE "-I" "" AOCL_COMPILE_CONFIG ${AOCL_COMPILE_CONFIG})
  STRING(REPLACE " " ";" AOCL_INCLUDE_DIRS ${AOCL_COMPILE_CONFIG})
  include_directories(${AOCL_INCLUDE_DIRS})

  execute_process(
    COMMAND aocl ldflags
    OUTPUT_VARIABLE AOCL_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  STRING(REPLACE "-L" "" AOCL_LDFLAGS ${AOCL_LDFLAGS})
  STRING(REPLACE " " ";" AOCL_LIB_DIRS ${AOCL_LDFLAGS})
  link_directorieS(${AOCL_LIB_DIRS})

  execute_process(
    COMMAND aocl ldlibs OUTPUT_VARIABLE AOCL_LDLIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  STRING(REPLACE "-l" "" AOCL_LDLIBS ${AOCL_LDLIBS})
  STRING(REPLACE " " ";" AOCL_LIBRARIES ${AOCL_LDLIBS})

  add_executable(cl-hash-fpga cl-hash.cc)
  target_link_libraries(cl-hash-fpga ${AOCL_LIBRARIES})
endif()

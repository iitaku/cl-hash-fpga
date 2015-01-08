# Global settings
project(cl-hash)
cmake_minimum_required(VERSION 2.8)

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

cmake_minimum_required(VERSION 3.15)
project(LoopRLOpt LANGUAGES CXX)

# Locate LLVM package
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to use" FORCE)

# Include LLVM helper functions
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Add the pass as a shared library
add_library(LoopRLOpt MODULE LoopRLOpt.cpp)

# Disable RTTI and exceptions if needed
target_compile_options(LoopRLOpt PRIVATE -fno-rtti -fno-exceptions)

# Include LLVM headers
target_include_directories(LoopRLOpt PRIVATE ${LLVM_INCLUDE_DIRS})
llvm_update_compile_flags(LoopRLOpt)

# Link LLVM components
llvm_map_components_to_libnames(LLVM_LIBS core support irreader passes)
target_link_libraries(LoopRLOpt PRIVATE ${LLVM_LIBS})

# Set the output prefix to empty (so the pass is named LoopRLOpt.so / .dylib / .dll)
set_target_properties(LoopRLOpt PROPERTIES PREFIX "")

# Optional: Enable warnings
target_compile_options(LoopRLOpt PRIVATE -Wall -Wextra -Wpedantic)

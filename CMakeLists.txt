cmake_minimum_required(VERSION 3.5.1)
project(celerity_std LANGUAGES CXX)
set(celerity_std_VERSION 0.1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Celerity REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

if(MSVC)
  # Add includes to library so they show up in generated VS project
  file(GLOB_RECURSE INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja")
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
endif()

add_library(celerity_std INTERFACE)

target_include_directories(
    celerity_std
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/celerity-std>)

target_link_libraries(
    celerity_std
    INTERFACE
    $<BUILD_INTERFACE:Celerity::celerity_runtime>
)

set(TARGET celerity_std CMAKE_CXX_STANDARD 17)
set(TARGET celerity_std CMAKE_CXX_STANDARD_REQUIRED ON)

#Examples

option(CELERITY_STD_BUILD_EXAMPLES "Build example applications" ON)

if(CELERITY_STD_BUILD_EXAMPLES)
    add_subdirectory(examples/matmul)
    add_subdirectory(examples/wave_sim)
    add_subdirectory(examples/convolution)

    set_property(
        TARGET convolution matmul wave_sim
        PROPERTY FOLDER "examples"
    )
endif()

# Tests

include(CTest)
enable_testing(true)
add_subdirectory(tests)

# Install

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include/celerity-std/celerity-std
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/celerity_std-config-version.cmake"
  VERSION ${celerity_std_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  TARGETS celerity_std
  EXPORT install_exports)

install(EXPORT install_exports
        FILE celerity_std-targets.cmake
        NAMESPACE Celerity::
        DESTINATION lib/cmake)

set(CELERITY_STD_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/celerity_std-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/celerity_std-config.cmake
  @ONLY
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/celerity_std-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/celerity_std-config-version.cmake
  DESTINATION lib/cmake
)

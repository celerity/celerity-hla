find_package(Boost 1.65.0 COMPONENTS graph REQUIRED)
find_package(MPI 2.0 REQUIRED)

add_executable(
  convolution
  convolution.cc
)

set_property(TARGET convolution PROPERTY CXX_STANDARD 17)

target_link_libraries(convolution
	PUBLIC
	Boost::boost
	MPI::MPI_CXX)

#add_celerity_to_target(
#  TARGET sequences
#  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/sequences.cpp
#)

if(MSVC)
  target_compile_options(convolution PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(convolution PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
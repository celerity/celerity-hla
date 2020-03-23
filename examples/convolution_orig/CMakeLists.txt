add_executable(
  convolution_orig
  convolution_orig.cc
)

set_property(TARGET convolution_orig PROPERTY CXX_STANDARD 14)

target_link_libraries(convolution_orig
	PUBLIC
  Celerity::celerity_runtime
	Boost::boost
	MPI::MPI_CXX)

add_celerity_to_target(
  TARGET  convolution_orig
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/convolution_orig.cc
)

if(MSVC)
  target_compile_options(convolution_orig PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(convolution_orig PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

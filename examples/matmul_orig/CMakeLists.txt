add_executable(
  matmul_orig
  matmul.cc
)

set_property(TARGET matmul_orig PROPERTY CXX_STANDARD 14)

target_link_libraries(matmul_orig
	PUBLIC
  Celerity::celerity_runtime
	Boost::boost
	MPI::MPI_CXX)

add_celerity_to_target(
  TARGET matmul_orig
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cc
)

if(MSVC)
  target_compile_options(matmul_orig PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(matmul_orig PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

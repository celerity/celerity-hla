add_executable(
  simple
  simple.cc
)

target_link_libraries(simple
	PUBLIC
  Celerity::celerity_runtime
	Boost::boost
	MPI::MPI_CXX)

add_celerity_to_target(
  TARGET simple
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/simple.cc
)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET simple PROPERTY CXX_STANDARD 17)

if(MSVC)
  target_compile_options(simple PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(simple PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

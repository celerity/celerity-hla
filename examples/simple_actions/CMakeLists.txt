add_executable(
  simple_actions
  simple_actions.cc
)

set_property(TARGET simple_actions PROPERTY CXX_STANDARD 17)

target_link_libraries(simple_actions
	PUBLIC
  Celerity::celerity_runtime
	Boost::boost
	MPI::MPI_CXX)

add_celerity_to_target(
  TARGET simple_actions
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/simple_actions.cc
)

if(MSVC)
  target_compile_options(simple_actions PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(simple_actions PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
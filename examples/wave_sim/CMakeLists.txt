add_executable(
  wave_sim
  wave_sim.cc
)

set_property(TARGET wave_sim PROPERTY CXX_STANDARD 17)

target_link_libraries(wave_sim
	PUBLIC
  Celerity::celerity_runtime
	Boost::boost
	MPI::MPI_CXX)

add_celerity_to_target(
  TARGET wave_sim
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/wave_sim.cc
)

if(MSVC)
  target_compile_options(wave_sim PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(wave_sim PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

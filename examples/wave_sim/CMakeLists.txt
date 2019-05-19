add_executable(
  wave_sim
  wave_sim.cc
)

set_property(TARGET wave_sim PROPERTY CXX_STANDARD 14)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED)
find_package(MPI 2.0 REQUIRED)

target_link_libraries(wave_sim
	PUBLIC
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

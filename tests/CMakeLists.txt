function(add_tests TESTS)
    foreach(ONE_TEST_CPP ${ARGV})

        set(ONE_TEST_CPP ${CMAKE_CURRENT_SOURCE_DIR}/${ONE_TEST_CPP})
        get_filename_component(ONE_TEST_EXEC ${ONE_TEST_CPP} NAME_WE)
        get_filename_component(ONE_TEST_EXEC_DIR ${ONE_TEST_CPP} DIRECTORY)

        set(TARGET_NAME test_${ONE_TEST_EXEC})

        add_executable(${TARGET_NAME} ${ONE_TEST_CPP})
        set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_TEST_EXEC}) 
        set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD 20)

        target_link_libraries(
            ${TARGET_NAME} 
            PUBLIC
            Celerity::celerity_runtime
        )	

        add_celerity_to_target(
            TARGET ${TARGET_NAME} 
            SOURCES ${ONE_TEST_CPP}
        )

        if(MSVC)
            target_compile_options(${TARGET_NAME} PRIVATE /D_CRT_SECURE_NO_WARNINGS /MP /W3)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter)
        endif()

        add_test(NAME ${TARGET_NAME}
                COMMAND ${TARGET_NAME})
    endforeach()
endfunction()

add_tests(accessors.cpp
          algorithms.cpp
          index_utils.cpp
          traits.cpp
          fusion.cpp
          sequencing.cpp
          subranges.cpp)
          #fusion_disabling.cpp)

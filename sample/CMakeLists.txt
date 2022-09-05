set(SAMPLES_no_inputs
parse_expr
pause
walk_the_line
)

macro (build sample)
    add_executable(${sample} ${sample}.cpp)
    target_compile_options(${sample} PRIVATE ${BASE_COMPILE_FLAGS})
    target_link_libraries(${sample} PRIVATE hspp)
    set_target_properties(${sample} PROPERTIES CXX_EXTENSIONS OFF)
endmacro (build)

foreach(sample ${SAMPLES_no_inputs})
    build(${sample})
    add_test(${sample} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${sample})
endforeach()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_tests_properties(pause PROPERTIES PASS_REGULAR_EXPRESSION "")
else()
set_tests_properties(pause PROPERTIES PASS_REGULAR_EXPRESSION
"Step 1
Step 2
=== should print through step 2 ===
Step 3
Step 3
Step 1
Step 2
Step 3")
endif()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
build(coroutine)
add_test(coroutine ${PROJECT_SOURCE_DIR}/sample/coroutine.sh)
set_tests_properties(coroutine PROPERTIES PASS_REGULAR_EXPRESSION
"What's your name. 
Hello, Bob
What's your favorite color. 
I like Red, too.")
endif()
set(hspp_SAMPLES
parse_expr
pause
)

foreach(sample ${hspp_SAMPLES})
    add_executable(${sample} ${sample}.cpp)
    target_compile_options(${sample} PRIVATE ${BASE_COMPILE_FLAGS})
    target_link_libraries(${sample} PRIVATE hspp)
    set_target_properties(${sample} PROPERTIES CXX_EXTENSIONS OFF)
    add_test(${sample} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${sample})
endforeach()

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set_tests_properties(pause PROPERTIES PASS_REGULAR_EXPRESSION "")
else()
set_tests_properties(pause PROPERTIES PASS_REGULAR_EXPRESSION
"Let's begin
Step 1
Step 2
=== should print through step 2 ===
Step 3
Yay, we're done!
Step 3
Yay, we're done!
Let's begin
Step 1
Step 2
Step 3
Yay, we're done!")
endif()

set(hspp_SAMPLES
parse_expr
)

foreach(sample ${hspp_SAMPLES})
    add_executable(${sample} ${sample}.cpp)
    target_compile_options(${sample} PRIVATE ${BASE_COMPILE_FLAGS})
    target_link_libraries(${sample} PRIVATE hspp)
    set_target_properties(${sample} PROPERTIES CXX_EXTENSIONS OFF)
    add_test(${sample} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${sample})
endforeach()
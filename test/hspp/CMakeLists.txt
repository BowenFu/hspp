add_executable(unittests test.cpp concurrent.cpp monadTrans.cpp)
target_include_directories(unittests PRIVATE)
target_link_libraries(unittests PRIVATE hspp gtest_main ${BASE_ADDITIONAL_LIBS})
set_target_properties(unittests PROPERTIES CXX_EXTENSIONS OFF)
gtest_discover_tests(unittests)
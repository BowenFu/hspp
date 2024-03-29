cmake_minimum_required(VERSION 3.15...3.19)

project(
    hspp
    VERSION 0.0.1
    LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_COLOR_MAKEFILE on)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

message(STATUS "CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

list(APPEND
    BASE_COMPILE_FLAGS
    "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-Wall;-Wextra;-pedantic;-Werror;-Wno-parentheses;-Wno-shadow;-Wconversion;-Wsign-conversion>"
        "$<$<CXX_COMPILER_ID:MSVC>:/W4>") # /WX for -Werror

list(APPEND
    BASE_COMPILE_FLAGS
    "$<$<CXX_COMPILER_ID:Clang>:-DFOR_CLANG>"
    "$<$<CXX_COMPILER_ID:AppleClang>:-DFOR_APPLECLANG>"
    "$<$<CXX_COMPILER_ID:GNU>:-DFOR_GNU;-Wno-maybe-uninitialized>"
    "$<$<CXX_COMPILER_ID:MSVC>:-DFOR_MSVC>"
    )

# Fix atomic lib linking for gcc.
list(APPEND
    BASE_ADDITIONAL_LIBS
    "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:atomic>")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    include(CodeCoverage)
    list(APPEND BASE_COMPILE_FLAGS "-g;-O0;-fprofile-arcs;-ftest-coverage")
endif() #CMAKE_BUILD_TYPE STREQUAL "Coverage"

if (CMAKE_BUILD_TYPE STREQUAL "MSAN")
    add_link_options("-L${PROJECT_SOURCE_DIR}/libcxx_msan/lib;-lc++abi")
endif() #CMAKE_BUILD_TYPE STREQUAL "MSAN"

# Target.
add_library(hspp INTERFACE)
target_include_directories(hspp INTERFACE 
  ${PROJECT_SOURCE_DIR}/include)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        include(Sanitizers)
        add_subdirectory(test)
        add_subdirectory(sample)
    endif()
endif()

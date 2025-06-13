cmake_minimum_required(VERSION 3.14)
project(methopts CXX)
set(CMAKE_CXX_STANDARD 20)

include(GoogleTest)

set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

enable_testing()

add_subdirectory(src/common)
add_subdirectory(task0_simplex)
add_subdirectory(task1_sgd_constrained)
add_subdirectory(task2_newton)
add_subdirectory(task3_advanced_sgd)
add_subdirectory(task4_lbfgs)
add_subdirectory(task5_branch_and_cut_tsp)
add_library(simplex src/Simplex.cpp)
target_include_directories(simplex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include)
find_package(Eigen3 REQUIRED)
target_link_libraries(simplex PUBLIC Eigen3::Eigen)

add_executable(test_simplex tests/test_simplex.cpp)
target_link_libraries(test_simplex PRIVATE simplex GTest::gtest GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_simplex)

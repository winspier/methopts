add_library(newton_optimizer src/NewtonOptimizer.cpp)
target_include_directories(newton_optimizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(newton_optimizer PUBLIC common)

add_executable(test_newton tests/test_newton.cpp)
target_link_libraries(test_newton PRIVATE newton_optimizer GTest::gtest GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_newton)

add_executable(train_constrained_newton train/train_constrained_newton.cpp)
target_link_libraries(train_constrained_newton PRIVATE newton_optimizer)

add_custom_target(demo_newton
        COMMAND train_constrained_newton > ${OUTPUT_DIR}/newton_result.txt
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/plot_lagrangian.py -2 2 -2 2 0.0
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS train_constrained_newton
        COMMENT "Run Newton method and plot result"
)

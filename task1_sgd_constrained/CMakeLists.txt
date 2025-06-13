add_library(constrained_sgd src/ConstrainedSGD.cpp src/LinearRegressionSGD.cpp)
target_include_directories(constrained_sgd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(constrained_sgd PUBLIC Threads::Threads)

add_executable(test_constrained_sgd tests/test_constrained_sgd.cpp)
target_link_libraries(test_constrained_sgd PRIVATE constrained_sgd GTest::gtest GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_constrained_sgd)

add_executable(test_linear_regression tests/test_linear_regression.cpp)
target_link_libraries(test_linear_regression PRIVATE constrained_sgd GTest::gtest GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_linear_regression)

add_executable(train_linear_regression train/train_linear_regression.cpp)
target_link_libraries(train_linear_regression PRIVATE constrained_sgd)

add_custom_target(train_and_plot_regression
        DEPENDS train_linear_regression
        COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/scripts/train_and_plot_regression.py
        dataset.tsv
        $<TARGET_FILE_DIR:train_linear_regression>
        ${OUTPUT_DIR}
        COMMENT "Train linear regression and plot results"
        USES_TERMINAL
)

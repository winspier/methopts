add_library(lbfgs src/LBFGS.cpp)
target_include_directories(lbfgs PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(test_lbfgs tests/test_lbfgs.cpp)
target_link_libraries(test_lbfgs PRIVATE
        lbfgs
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)
gtest_discover_tests(test_lbfgs)

add_executable(train_lbfgs train/train_lbfgs.cpp)
target_link_libraries(train_lbfgs PRIVATE lbfgs)

add_custom_target(run_lbfgs_with_history
        COMMAND train_lbfgs 100 > ${OUTPUT_DIR}/lbfgs_history.csv
        DEPENDS train_lbfgs
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Run L-BFGS and record history"
)

add_custom_target(plot_lbfgs
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/plot_lbfgs.py ${OUTPUT_DIR}/lbfgs_history.csv
        DEPENDS run_lbfgs_with_history
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Plot L-BFGS history"
)

add_executable(print_solution train/print_solution.cpp)
target_link_libraries(print_solution PRIVATE lbfgs)

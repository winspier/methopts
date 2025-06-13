add_library(optimizers src/Optimizers.cpp)
target_include_directories(optimizers PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(test_optimizers tests/test_optimizers.cpp)
target_link_libraries(test_optimizers PRIVATE
        optimizers
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)
gtest_discover_tests(test_optimizers)

add_executable(run_experiment train/experiments.cpp)
target_link_libraries(run_experiment PRIVATE optimizers)

set(CONVERGENCE_CSV ${OUTPUT_DIR}/convergence.csv)

add_custom_command(
        OUTPUT ${CONVERGENCE_CSV}
        COMMAND run_experiment ${CONVERGENCE_CSV}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Run optimizer convergence experiment and write to ${CONVERGENCE_CSV}"
        VERBATIM
)

add_custom_target(run_experiments ALL
        DEPENDS ${CONVERGENCE_CSV}
)

add_custom_target(plot_convergence
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/plot_convergence.py ${CONVERGENCE_CSV}
        DEPENDS ${CONVERGENCE_CSV}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Plot optimizer convergence"
)

add_custom_target(show_convergence
    DEPENDS run_experiment
    COMMAND $<TARGET_FILE:run_experiment> ${CONVERGENCE_CSV}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Run run_experiment and print final convergence points"
    USES_TERMINAL
)


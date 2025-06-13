add_library(tsp_solver src/BranchAndCutSolver.cpp src/Graph.cpp src/LPModel.cpp)
target_include_directories(tsp_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tsp_solver PUBLIC common simplex)
target_compile_options(tsp_solver PRIVATE -Wno-multichar)

add_executable(test_tsp tests/test_branch_and_cut.cpp)
target_link_libraries(test_tsp PRIVATE tsp_solver GTest::gtest GTest::gtest_main Threads::Threads)
gtest_discover_tests(test_tsp)

add_executable(solve_tsp train/demo_bac_tsp.cpp)
find_package(nlohmann_json REQUIRED)
target_link_libraries(solve_tsp PRIVATE tsp_solver nlohmann_json::nlohmann_json)

add_custom_target(plot_tsp
        COMMAND solve_tsp > ${OUTPUT_DIR}/solution.json
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/plot_tsp.py ${OUTPUT_DIR}/solution.json ${OUTPUT_DIR}/tsp_plot.png
        DEPENDS solve_tsp
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Solve TSP and plot tour"
)

#include <gtest/gtest.h>
#include "Simplex.h"

static constexpr double EPS = 1e-6;

TEST(SimplexTest, SimpleCase) {
    std::vector<std::vector<double>> A = {
        {1, 1},
        {1, 0},
        {0, 1}
    };
    std::vector<double> b = {4, 2, 3};
    std::vector<double> c = {3, 2};

    Simplex solver(A, b, c);
    std::vector<double> solution;
    double result = solver.solve(solution);

    EXPECT_NEAR(result, 10.0, EPS);
    EXPECT_NEAR(solution[0], 2.0, EPS);
    EXPECT_NEAR(solution[1], 2.0, EPS);
}

TEST(SimplexTest, Unbounded) {
    std::vector<std::vector<double>> A = {
        {-1, 1},
        {0, -1}
    };
    std::vector<double> b = {-1, 0};
    std::vector<double> c = {1, 1};

    Simplex solver(A, b, c);
    std::vector<double> solution;
    double result = solver.solve(solution);

    EXPECT_EQ(result, std::numeric_limits<double>::infinity());
}

TEST(SimplexTest, SingleConstraint) {
    std::vector<std::vector<double>> A = {
        {1}
    };
    std::vector<double> b = {5};
    std::vector<double> c = {1};

    Simplex solver(A, b, c);
    std::vector<double> solution;
    double result = solver.solve(solution);

    EXPECT_NEAR(result, 5.0, EPS);
    ASSERT_EQ(solution.size(), 1);
    EXPECT_NEAR(solution[0], 5.0, EPS);
}

TEST(SimplexTest, ZeroObjective) {
    std::vector<std::vector<double>> A = {
        {1, 1}
    };
    std::vector<double> b = {5};
    std::vector<double> c = {0, 0};

    Simplex solver(A, b, c);
    std::vector<double> solution;
    double result = solver.solve(solution);

    EXPECT_NEAR(result, 0.0, EPS);
}

TEST(SimplexTest, MultipleOptimal) {
    std::vector<std::vector<double>> A = {
        {1, 1}
    };
    std::vector<double> b = {1};
    std::vector<double> c = {1, 1};

    Simplex solver(A, b, c);
    std::vector<double> solution;
    double result = solver.solve(solution);

    EXPECT_NEAR(result, 1.0, EPS);
    ASSERT_EQ(solution.size(), 2);
    EXPECT_NEAR(solution[0] + solution[1], 1.0, EPS);
    EXPECT_GE(solution[0], -EPS);
    EXPECT_GE(solution[1], -EPS);
}

TEST(SimplexTest, LargerDimension) {
    std::vector<std::vector<double>> A = {
        {1, 1, 1},
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<double> b = {7, 3, 4, 5};
    std::vector<double> c = {4, 3, 2};

    Simplex solver(A, b, c);
    std::vector<double> solution;
    double result = solver.solve(solution);

    EXPECT_NEAR(result, 4*3 + 3*4 + 2*0, EPS);
    ASSERT_EQ(solution.size(), 3);
    EXPECT_NEAR(solution[0], 3.0, EPS);
    EXPECT_NEAR(solution[1], 4.0, EPS);
    EXPECT_NEAR(solution[2], 0.0, EPS);
}



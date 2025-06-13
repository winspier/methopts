#include <gtest/gtest.h>
#include "LinearRegressionSGD.h"

TEST(LinearRegressionTest, SimpleLine) {
    std::vector<Vec> X = {{1,2}, {2,1}, {3,0}};
    Vec y = {2*1+3*2, 2*2+3*1, 2*3+3*0};
    Vec lower = {-10, -10};
    Vec upper = {10, 10};
    LinearRegressionSGD lr(0.01, 10000, lower, upper);
    Vec beta0 = {0,0};
    Vec beta = lr.fit(X, y, beta0);
    EXPECT_NEAR(beta[0], 2.0, 1e-1);
    EXPECT_NEAR(beta[1], 3.0, 1e-1);
}

TEST(LinearRegressionTest, ConstantOutput) {
    std::vector<Vec> X = {{1, 1}, {1, 1}, {1, 1}};
    Vec y = {5, 5, 5};
    Vec lower = {-10, -10};
    Vec upper = {10, 10};
    LinearRegressionSGD lr(0.01, 5000, lower, upper);
    Vec beta0 = {0, 0};
    Vec beta = lr.fit(X, y, beta0);
    double prediction = beta[0] * 1 + beta[1] * 1;
    EXPECT_NEAR(prediction, 5.0, 1e-1);
}

TEST(LinearRegressionTest, ZeroTarget) {
    std::vector<Vec> X = {{1, 2}, {3, 4}, {5, 6}};
    Vec y = {0, 0, 0};
    Vec lower = {-10, -10};
    Vec upper = {10, 10};
    LinearRegressionSGD lr(0.01, 10000, lower, upper);
    Vec beta0 = {1, 1};
    Vec beta = lr.fit(X, y, beta0);
    EXPECT_NEAR(beta[0], 0.0, 1e-1);
    EXPECT_NEAR(beta[1], 0.0, 1e-1);
}

TEST(LinearRegressionTest, BoundsEnforced) {
    std::vector<Vec> X = {{10, 10}, {20, 20}, {30, 30}};
    Vec y = {1000, 2000, 3000};

    Vec lower = {0.0, 0.0};
    Vec upper = {1.0, 1.0};
    LinearRegressionSGD lr(0.1, 1000, lower, upper);
    Vec beta0 = {0.5, 0.5};
    Vec beta = lr.fit(X, y, beta0);
    EXPECT_LE(beta[0], 1.0);
    EXPECT_GE(beta[0], 0.0);
    EXPECT_LE(beta[1], 1.0);
    EXPECT_GE(beta[1], 0.0);
}

TEST(LinearRegressionTest, PerfectFit) {
    std::vector<Vec> X = {{1, 0}, {0, 1}, {2, 3}};
    Vec y = {4, 5, 4*2 + 5*3};
    Vec lower = {-10, -10};
    Vec upper = {10, 10};
    LinearRegressionSGD lr(0.01, 10000, lower, upper);
    Vec beta0 = {0, 0};
    Vec beta = lr.fit(X, y, beta0);
    EXPECT_NEAR(beta[0], 4.0, 1e-1);
    EXPECT_NEAR(beta[1], 5.0, 1e-1);
}



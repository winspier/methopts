#include <gtest/gtest.h>
#include "NewtonOptimizer.h"

TEST(NewtonOptimizerTest, QuadraticFunction) {
    auto f = [](const Vec& x) {
        return (x[0] - 1)*(x[0] - 1) + (x[1] + 2)*(x[1] + 2);
    };
    auto grad = [](const Vec& x) {
        return Vec{2 * (x[0] - 1), 2 * (x[1] + 2)};
    };
    auto hess = [](const Vec& x) {
        return std::vector<Vec>{
                {2.0, 0.0},
                {0.0, 2.0}
        };
    };

    NewtonOptimizer opt(1e-6, 100);
    Vec x0 = {0.0, 0.0};
    Vec result = opt.optimize(f, grad, hess, x0);
    EXPECT_NEAR(result[0], 1.0, 1e-4);
    EXPECT_NEAR(result[1], -2.0, 1e-4);
}

TEST(NewtonOptimizerTest, OriginMinimum) {
    auto f = [](const Vec& x) {
        return x[0]*x[0] + x[1]*x[1];
    };
    auto grad = [](const Vec& x) {
        return Vec{2*x[0], 2*x[1]};
    };
    auto hess = [](const Vec& x) {
        return std::vector<Vec>{
                {2.0, 0.0},
                {0.0, 2.0}
        };
    };

    NewtonOptimizer opt(1e-6, 50);
    Vec x0 = {3.0, -4.0};
    Vec result = opt.optimize(f, grad, hess, x0);
    EXPECT_NEAR(result[0], 0.0, 1e-4);
    EXPECT_NEAR(result[1], 0.0, 1e-4);
}

TEST(NewtonOptimizerTest, ExponentialFunction) {
    auto f = [](const Vec& x) {
        return std::exp(x[0]) - x[0];
    };
    auto grad = [](const Vec& x) {
        return Vec{std::exp(x[0]) - 1.0};
    };
    auto hess = [](const Vec& x) {
        return std::vector<Vec>{{std::exp(x[0])}};
    };

    NewtonOptimizer opt(1e-6, 100);
    Vec x0 = {2.0};
    Vec result = opt.optimize(f, grad, hess, x0);
    EXPECT_NEAR(result[0], 0.0, 1e-4);
}

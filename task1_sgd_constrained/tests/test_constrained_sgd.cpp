#include <gtest/gtest.h>

#include "ConstrainedSGD.h"

TEST(ConstrainedSGDTest, QuadraticConstrained) {
    auto f = [](const Vec& v) {
        return (v[0] - 1.0)*(v[0] - 1.0) + (v[1] + 2.0)*(v[1] + 2.0);
    };
    auto grad = [](const Vec& v) {
        return Vec{2*(v[0] - 1.0), 2*(v[1] + 2.0)};
    };
    Vec lower = {0.0, -1.0};
    Vec upper = {2.0, 2.0};
    ConstrainedSGD solver(0.1, 1000, lower, upper);
    Vec x0 = {5.0, 5.0};
    Vec result = solver.optimize(f, grad, x0);
    EXPECT_NEAR(result[0], 1.0, 1e-3);
    EXPECT_NEAR(result[1], -1.0, 1e-3);
}

TEST(ConstrainedSGDTest, QuadraticUnconstrained) {
    auto f = [](const Vec& v) {
        return (v[0] - 3.0)*(v[0] - 3.0) + (v[1] - 4.0)*(v[1] - 4.0);
    };
    auto grad = [](const Vec& v) {
        return Vec{2*(v[0] - 3.0), 2*(v[1] - 4.0)};
    };
    Vec lower = {-10.0, -10.0};
    Vec upper = {10.0, 10.0};
    ConstrainedSGD solver(0.05, 500, lower, upper);
    Vec x0 = {0.0, 0.0};
    Vec result = solver.optimize(f, grad, x0);
    EXPECT_NEAR(result[0], 3.0, 1e-3);
    EXPECT_NEAR(result[1], 4.0, 1e-3);
}

TEST(ConstrainedSGDTest, StartingPointInsideBounds) {
    auto f = [](const Vec& v) {
        return v[0]*v[0] + v[1]*v[1];
    };
    auto grad = [](const Vec& v) {
        return Vec{2*v[0], 2*v[1]};
    };
    Vec lower = {1.0, 1.0};
    Vec upper = {5.0, 5.0};
    ConstrainedSGD solver(0.1, 100, lower, upper);
    Vec x0 = {2.0, 3.0};
    Vec result = solver.optimize(f, grad, x0);
    EXPECT_NEAR(result[0], 1.0, 1e-3);
    EXPECT_NEAR(result[1], 1.0, 1e-3);
}

TEST(ConstrainedSGDTest, StartOutsideBounds) {
    auto f = [](const Vec& v) {
        return (v[0] - 2.0)*(v[0] - 2.0) + (v[1] - 3.0)*(v[1] - 3.0);
    };
    auto grad = [](const Vec& v) {
        return Vec{2*(v[0] - 2.0), 2*(v[1] - 3.0)};
    };
    Vec lower = {0.0, 0.0};
    Vec upper = {4.0, 4.0};
    ConstrainedSGD solver(0.1, 500, lower, upper);
    Vec x0 = {10.0, -5.0};  // вне границ
    Vec result = solver.optimize(f, grad, x0);
    EXPECT_NEAR(result[0], 2.0, 1e-2);
    EXPECT_NEAR(result[1], 3.0, 1e-2);
}

TEST(ConstrainedSGDTest, MinimumOutsideBounds) {
    auto f = [](const Vec& v) {
        return (v[0] + 5.0)*(v[0] + 5.0) + (v[1] + 5.0)*(v[1] + 5.0);
    };
    auto grad = [](const Vec& v) {
        return Vec{2*(v[0] + 5.0), 2*(v[1] + 5.0)};
    };
    Vec lower = {-2.0, -2.0};
    Vec upper = {2.0, 2.0};
    ConstrainedSGD solver(0.05, 1000, lower, upper);
    Vec x0 = {0.0, 0.0};
    Vec result = solver.optimize(f, grad, x0);
    EXPECT_NEAR(result[0], -2.0, 1e-3);
    EXPECT_NEAR(result[1], -2.0, 1e-3);
}

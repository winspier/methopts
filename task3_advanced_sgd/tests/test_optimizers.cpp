#include <gtest/gtest.h>
#include "Optimizers.h"

double f_quad(const Vec& x) { return x[0]*x[0]; }
Vec grad_quad(const Vec& x) { return Vec{2*x[0]}; }
double f_quad_2d(const Vec& x) { return (x[0] - 3) * (x[0] - 3) + (x[1] + 1) * (x[1] + 1); }
Vec grad_quad_2d(const Vec& x) { return Vec{2 * (x[0] - 3), 2 * (x[1] + 1)}; }

TEST(OptimizersTest, Quadratic2D_DecreasesFunction) {
    Vec x0 = {0.0, 0.0};
    int max_iters = 1000;
    double f0 = f_quad_2d(x0);

    GradientDescent gd(1e-2, max_iters);
    OptimizerResult r = gd.optimize(f_quad_2d, grad_quad_2d, x0);

    double ffinal = r.history.back();
    EXPECT_LT(ffinal, f0) << "GD did not decrease function value in 2D quadratic";

    EXPECT_NEAR(r.x[0], 3.0, 1e-2);
    EXPECT_NEAR(r.x[1], -1.0, 1e-2);
}

double f_constant(const Vec& x) { return 42.0; }
Vec grad_zero(const Vec& x) { return Vec(x.size(), 0.0); }

TEST(OptimizersTest, ConstantFunction_ZeroGradient) {
    Vec x0 = {1.0, -2.0, 3.0};
    int max_iters = 100;

    GradientDescent gd(1e-1, max_iters);
    OptimizerResult r = gd.optimize(f_constant, grad_zero, x0);

    for (double v : r.history) {
        EXPECT_DOUBLE_EQ(v, 42.0);
    }

    for (size_t i = 0; i < x0.size(); ++i) {
        EXPECT_DOUBLE_EQ(r.x[i], x0[i]);
    }
}

TEST(OptimizersTest, HistoryIsCorrectLength) {
    Vec x0 = {10.0};
    int max_iters = 50;
    GradientDescent gd(1e-2, max_iters);
    OptimizerResult r = gd.optimize(f_quad, grad_quad, x0);

    EXPECT_EQ(r.history.size(), max_iters + 1);
}


TEST(OptimizersTest, Quadratic1D_DecreasesFunction) {
    Vec x0 = {5.0};
    int max_iters = 1000;
    double f0 = f_quad(x0);
    std::vector lrs = {1e-3, 1e-2, 1e-1};
    for(double lr : lrs) {
        {
            GradientDescent gd(lr, max_iters);
            OptimizerResult r = gd.optimize(f_quad, grad_quad, x0);
            double ffinal = r.history.back();
            EXPECT_LT(ffinal, f0) << "GD with lr=" << lr << " did not decrease f: f0=" << f0 << ", ffinal=" << ffinal;
            for(size_t i = 1; i < r.history.size(); ++i) {
                EXPECT_LE(r.history[i], r.history[i-1] + 1e-12)
                    << "GD history not non-increasing at iter " << i << " for lr=" << lr
                    << ": prev=" << r.history[i-1] << ", curr=" << r.history[i];
            }
        }
        {
            MomentumGD mgd(lr, max_iters, 0.9);
            OptimizerResult r = mgd.optimize(f_quad, grad_quad, x0);
            double ffinal = r.history.back();
            EXPECT_LT(ffinal, f0) << "MomentumGD with lr=" << lr << " did not decrease f: f0=" << f0 << ", ffinal=" << ffinal;
            double fmin = r.history[0];
            for(double v : r.history) {
                if(v < fmin) fmin = v;
            }
            EXPECT_LT(fmin, f0) << "MomentumGD with lr=" << lr << " never achieved f < f0; f0=" << f0;
        }
        {
            AdamOptimizer adam(lr, max_iters, 0.9, 0.999, 1e-8);
            OptimizerResult r = adam.optimize(f_quad, grad_quad, x0);
            double ffinal = r.history.back();
            EXPECT_LT(ffinal, f0) << "Adam with lr=" << lr << " did not decrease f: f0=" << f0 << ", ffinal=" << ffinal;
            double fmin = r.history[0];
            for(double v : r.history) {
                if(v < fmin) fmin = v;
            }
            EXPECT_LT(fmin, f0) << "Adam with lr=" << lr << " never achieved f < f0; f0=" << f0;
        }
    }
}

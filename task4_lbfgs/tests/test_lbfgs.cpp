#include <gtest/gtest.h>
#include "LBFGS.h"

struct RosenbrockPairs {
    RosenbrockPairs(int N) : N_(N) {}

    double operator()(const Vec& x) const {
        double sum = 0;
        for (int i = 0; i+1 < N_; i += 2) {
            double t1 = x[i], t2 = x[i+1];
            sum += 100*(t1*t1 - t2)*(t1*t1 - t2)
                 + (t1 - 1)*(t1 - 1);
        }
        return sum;
    }

    Vec grad(const Vec& x) const {
        Vec g(N_, 0.0);
        for (int i = 0; i+1 < N_; i += 2) {
            double t1 = x[i], t2 = x[i+1];
            g[i]   = 400*t1*(t1*t1 - t2) + 2*(t1 - 1);
            g[i+1] = -200*(t1*t1 - t2);
        }
        return g;
    }

private:
    int N_;
};

TEST(LBFGSTest, RosenbrockN10) {
    const int N = 10;
    RosenbrockPairs problem(N);
    Vec x0(N, 0.0);
    int m = 5;
    int max_iters = 2000;
    double tol = 1e-6;
    LBFGS opt(m, max_iters, tol);

    Vec res = opt.optimize(
        [&](const Vec& x){ return problem(x); },
        [&](const Vec& x){ return problem.grad(x); },
        x0
    );

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(res[i], 1.0, 1e-3) << "at index " << i;
    }
}

TEST(LBFGSTest, RosenbrockN11OddSize) {
    const int N = 11;
    RosenbrockPairs problem(N);
    Vec x0(N, 0.0);
    LBFGS opt(5, 2000, 1e-6);

    Vec res = opt.optimize(
        [&](const Vec& x){ return problem(x); },
        [&](const Vec& x){ return problem.grad(x); },
        x0
    );

    for (int i = 0; i+1 < N; i += 2) {
        EXPECT_NEAR(res[i],   1.0, 1e-3) << "at paired index " << i;
        EXPECT_NEAR(res[i+1], 1.0, 1e-3) << "at paired index " << i+1;
    }
    EXPECT_NEAR(res[N-1], 0.0, 1e-8);
}

TEST(LBFGSTest, RosenbrockStartingNearMinimum) {
    const int N = 10;
    RosenbrockPairs problem(N);
    Vec x0(N, 1.0);
    LBFGS opt(5, 100, 1e-10);

    Vec res = opt.optimize(
        [&](const Vec& x){ return problem(x); },
        [&](const Vec& x){ return problem.grad(x); },
        x0
    );

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(res[i], 1.0, 1e-8) << "at index " << i;
    }
}

TEST(LBFGSTest, ConstantFunction_ZeroGradient) {
    const int N = 5;
    Vec x0(N, 3.14);
    LBFGS opt(3, 50, 1e-8);

    auto f_const = [](const Vec&){ return 42.0; };
    auto grad_zero = [](const Vec& x){ return Vec(x.size(), 0.0); };

    Vec res = opt.optimize(f_const, grad_zero, x0);

    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(res[i], x0[i]);
    }
}


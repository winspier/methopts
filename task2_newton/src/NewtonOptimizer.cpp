#include "NewtonOptimizer.h"

static Vec solve_linear_system(const std::vector<Vec>& H, const Vec& g) {
    const size_t n = g.size();
    std::vector A(H);
    Vec b = g;

    for (size_t i = 0; i < n; ++i) {
        double diag = A[i][i];
        for (size_t j = i; j < n; ++j) A[i][j] /= diag;
        b[i] /= diag;

        for (size_t k = i + 1; k < n; ++k) {
            double factor = A[k][i];
            for (size_t j = i; j < n; ++j)
                A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }
    }

    Vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (size_t j = i + 1; j < n; ++j)
            x[i] -= A[i][j] * x[j];
    }

    return x;
}

NewtonOptimizer::NewtonOptimizer(double tol, int max_iters)
    : tol_(tol), max_iters_(max_iters) {}

Vec NewtonOptimizer::optimize(
    const std::function<double(const Vec&)>& f,
    const std::function<Vec(const Vec&)>& grad,
    const std::function<std::vector<Vec>(const Vec&)>& hess,
    Vec x0
) const {
    Vec x = x0;
    for (int i = 0; i < max_iters_; ++i) {
        Vec g = grad(x);
        if (common::norm2(g) < tol_) break;

        auto H = hess(x);
        Vec delta = solve_linear_system(H, g);
        for (size_t j = 0; j < x.size(); ++j)
            x[j] -= delta[j];
    }
    return x;
}

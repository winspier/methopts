#include "LinearRegressionSGD.h"
#include "ConstrainedSGD.h"

LinearRegressionSGD::LinearRegressionSGD(double learning_rate, int max_iters,
                                         const Vec& lower_bounds,
                                         const Vec& upper_bounds)
    : lower_(lower_bounds), upper_(upper_bounds), lr_(learning_rate), max_iters_(max_iters) {}

Vec LinearRegressionSGD::fit(const std::vector<Vec>& X, const Vec& y, Vec beta0) const {
    size_t m = X.size();
    auto f = [&](const Vec& beta) {
        double loss = 0;
        for (size_t i = 0; i < m; ++i) {
            double pred = 0;
            for (size_t j = 0; j < beta.size(); ++j) pred += X[i][j] * beta[j];
            double diff = pred - y[i];
            loss += diff * diff;
        }
        return loss / (2 * m);
    };
    auto grad = [&](const Vec& beta) {
        Vec g(beta.size(), 0.0);
        for (size_t i = 0; i < m; ++i) {
            double pred = 0;
            for (size_t j = 0; j < beta.size(); ++j) pred += X[i][j] * beta[j];
            double diff = pred - y[i];
            for (size_t j = 0; j < beta.size(); ++j) {
                g[j] += diff * X[i][j];
            }
        }
        for (double& v : g) v = v / m;
        return g;
    };
    ConstrainedSGD sgd(lr_, max_iters_, lower_, upper_);
    return sgd.optimize(f, grad, beta0);
}

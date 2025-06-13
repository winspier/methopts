#include "ConstrainedSGD.h"
#include <algorithm>

ConstrainedSGD::ConstrainedSGD(double learning_rate, int max_iters,
                               const Vec& lower_bounds,
                               const Vec& upper_bounds)
    : lr_(learning_rate), max_iters_(max_iters),
      lower_(lower_bounds), upper_(upper_bounds) {}

Vec ConstrainedSGD::optimize(const std::function<double(const Vec&)>& f,
                             const std::function<Vec(const Vec&)>& grad,
                             Vec x0) const {
    Vec x = x0;
    for (int iter = 0; iter < max_iters_; ++iter) {
        Vec g = grad(x);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= lr_ * g[i];
        }
        project(x);
    }
    return x;
}

void ConstrainedSGD::project(Vec& x) const {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::min(std::max(x[i], lower_[i]), upper_[i]);
    }
}
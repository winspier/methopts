#pragma once
#include <vector>
#include <functional>
#include "common/Types.h"

class NewtonOptimizer {
public:
    NewtonOptimizer(double tol, int max_iters);

    Vec optimize(
        const std::function<double(const Vec&)>& f,
        const std::function<Vec(const Vec&)>& grad,
        const std::function<std::vector<Vec>(const Vec&)>& hess,
        Vec x0
    ) const;

private:
    double tol_;
    int max_iters_;
};

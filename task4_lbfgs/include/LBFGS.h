#pragma once
#include <vector>
#include <functional>
#include "common/Types.h"

using Function    = std::function<double(const Vec&)>;
using Gradient    = std::function<Vec(const Vec&)>;
using HistoryCB   = std::function<void(int iter, const Vec& x, double loss, double grad_norm)>;

class LBFGS {
public:
    LBFGS(int m, int max_iters = 1000, double tol = 1e-6);

    Vec optimize(const Function& f,
                 const Gradient& grad,
                 Vec& x,
                 HistoryCB history_cb);

    Vec optimize(const Function& f,
                 const Gradient& grad,
                 const Vec& x0);

private:
    int m_, max_iters_;
    double tol_;
};

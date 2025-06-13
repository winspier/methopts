#pragma once
#include <vector>
#include <functional>
#include "common/Types.h"

class ConstrainedSGD {
public:
    ConstrainedSGD(double learning_rate, int max_iters,
                   const Vec& lower_bounds, const Vec& upper_bounds);
    Vec optimize(const std::function<double(const Vec&)>& f,
                 const std::function<Vec(const Vec&)>& grad,
                 Vec x0) const;

private:
    void project(Vec& x) const;
    double lr_;
    int max_iters_;
    Vec lower_;
    Vec upper_;
};
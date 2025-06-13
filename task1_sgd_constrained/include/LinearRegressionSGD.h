#pragma once
#include <vector>
#include "common/Types.h"

class LinearRegressionSGD {
public:
    LinearRegressionSGD(double learning_rate, int max_iters,
                        const Vec& lower_bounds, const Vec& upper_bounds);
    Vec fit(const std::vector<Vec>& X, const Vec& y, Vec beta0) const;
private:
    double lr_;
    int max_iters_;
    Vec lower_;
    Vec upper_;

};
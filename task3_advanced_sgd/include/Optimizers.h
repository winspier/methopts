#pragma once
#include <vector>
#include <functional>

using Vec = std::vector<double>;

using Function = std::function<double(const Vec&)>;
using Gradient = std::function<Vec(const Vec&)>;

struct OptimizerResult {
    Vec x;
    std::vector<double> history;
};

class GradientDescent {
public:
    GradientDescent(double lr, int max_iters);
    OptimizerResult optimize(const Function& f, const Gradient& grad, Vec x0) const;

private:
    double lr_;
    int max_iters_;
};

class MomentumGD {
public:
    MomentumGD(double lr, int max_iters, double beta = 0.9);
    OptimizerResult optimize(const Function& f, const Gradient& grad, Vec x0) const;

private:
    double lr_;
    int max_iters_;
    double beta_;
};

class AdamOptimizer {
public:
    AdamOptimizer(double lr, int max_iters, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
    OptimizerResult optimize(const Function& f, const Gradient& grad, Vec x0) const;

private:
    double lr_;
    int max_iters_;
    double beta1_, beta2_, eps_;
};

#include "Optimizers.h"
#include <cmath>

GradientDescent::GradientDescent(double lr, int max_iters)
    : lr_(lr), max_iters_(max_iters) {}

OptimizerResult GradientDescent::optimize(const Function& f, const Gradient& grad, Vec x0) const {
    OptimizerResult res;
    res.x = x0;
    res.history.reserve(max_iters_ + 1);
    double fx = f(res.x);
    res.history.push_back(fx);
    for (int iter = 1; iter <= max_iters_; ++iter) {
        Vec g = grad(res.x);
        for (size_t i = 0; i < res.x.size(); ++i) {
            res.x[i] -= lr_ * g[i];
        }
        fx = f(res.x);
        res.history.push_back(fx);
        double norm2 = 0;
        for (double v : g) norm2 += v * v;
        if (std::sqrt(norm2) < 1e-8) break;
    }
    return res;
}

MomentumGD::MomentumGD(double lr, int max_iters, double beta)
    : lr_(lr), max_iters_(max_iters), beta_(beta) {}

OptimizerResult MomentumGD::optimize(const Function& f, const Gradient& grad, Vec x0) const {
    OptimizerResult res;
    res.x = x0;
    res.history.reserve(max_iters_ + 1);
    Vec v(x0.size(), 0.0);
    double fx = f(res.x);
    res.history.push_back(fx);
    for (int iter = 1; iter <= max_iters_; ++iter) {
        Vec g = grad(res.x);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = beta_ * v[i] + (1.0 - beta_) * g[i];
        }
        for (size_t i = 0; i < res.x.size(); ++i) {
            res.x[i] -= lr_ * v[i];
        }
        fx = f(res.x);
        res.history.push_back(fx);
        double norm2 = 0;
        for (double vv : v) norm2 += vv * vv;
        if (std::sqrt(norm2) < 1e-8) break;
    }
    return res;
}

AdamOptimizer::AdamOptimizer(double lr, int max_iters, double beta1, double beta2, double eps)
    : lr_(lr), max_iters_(max_iters), beta1_(beta1), beta2_(beta2), eps_(eps) {}

OptimizerResult AdamOptimizer::optimize(const Function& f, const Gradient& grad, Vec x0) const {
    OptimizerResult res;
    res.x = x0;
    res.history.reserve(max_iters_ + 1);
    Vec m(x0.size(), 0.0);
    Vec v(x0.size(), 0.0);
    double fx = f(res.x);
    res.history.push_back(fx);
    for (int iter = 1; iter <= max_iters_; ++iter) {
        Vec g = grad(res.x);
        for (size_t i = 0; i < g.size(); ++i) {
            m[i] = beta1_ * m[i] + (1.0 - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0 - beta2_) * g[i] * g[i];
        }
        double bias_correction1 = 1.0 - std::pow(beta1_, iter);
        double bias_correction2 = 1.0 - std::pow(beta2_, iter);
        for (size_t i = 0; i < res.x.size(); ++i) {
            double m_hat = m[i] / bias_correction1;
            double v_hat = v[i] / bias_correction2;
            res.x[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
        fx = f(res.x);
        res.history.push_back(fx);
        double norm2 = 0;
        for (double gi : g) norm2 += gi * gi;
        if (std::sqrt(norm2) < 1e-8) break;
    }
    return res;
}

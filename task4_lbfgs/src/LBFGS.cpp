#include "LBFGS.h"
#include <cmath>
#include <deque>
#include <numeric>

LBFGS::LBFGS(int m, int max_iters, double tol)
    : m_(m), max_iters_(max_iters), tol_(tol) {}

Vec LBFGS::optimize(const Function& f,
                    const Gradient& grad,
                    Vec& x,
                    HistoryCB history_cb) {
    const int n = x.size();
    std::deque<Vec> s_list, y_list;
    Vec q(n), alpha, rho;

    for (int iter = 0; iter < max_iters_; ++iter) {
        Vec g = grad(x);
        double grad_norm = std::sqrt(std::inner_product(g.begin(), g.end(), g.begin(), 0.0));
        double loss = f(x);
        if (history_cb) history_cb(iter, x, loss, grad_norm);
        if (grad_norm < tol_) break;

        q = g;
        int k = s_list.size();
        alpha.assign(k, 0.0);
        rho.assign(k, 0.0);

        for (int i = k-1; i >= 0; --i) {
            rho[i] = 1.0 / std::inner_product(y_list[i].begin(), y_list[i].end(),
                                             s_list[i].begin(), 0.0);
            alpha[i] = rho[i] * std::inner_product(s_list[i].begin(), s_list[i].end(),
                                                   q.begin(), 0.0);
            for (int j = 0; j < n; ++j) q[j] -= alpha[i] * y_list[i][j];
        }

        double gamma = 1.0;
        if (k > 0) {
            double sy = std::inner_product(s_list.back().begin(), s_list.back().end(),
                                           y_list.back().begin(), 0.0);
            double yy = std::inner_product(y_list.back().begin(), y_list.back().end(),
                                           y_list.back().begin(), 0.0);
            gamma = sy / yy;
        }
        for (int j = 0; j < n; ++j) q[j] *= gamma;

        Vec z = q;
        for (int i = 0; i < k; ++i) {
            double beta = rho[i] * std::inner_product(y_list[i].begin(), y_list[i].end(),
                                                      z.begin(), 0.0);
            for (int j = 0; j < n; ++j)
                z[j] += s_list[i][j] * (alpha[i] - beta);
        }
        for (int j = 0; j < n; ++j) z[j] = -z[j];

        double step = 1.0, c = 1e-4;
        double f0 = loss;
        Vec x_new(n);
        while (true) {
            for (int j = 0; j < n; ++j)
                x_new[j] = x[j] + step * z[j];
            double f1 = f(x_new);
            double dot = std::inner_product(g.begin(), g.end(), z.begin(), 0.0);
            if (f1 <= f0 + c * step * dot) break;
            step *= 0.5;
            if (step < 1e-20) break;
        }

        Vec s(n), yv(n);
        Vec g_new = grad(x_new);
        for (int j = 0; j < n; ++j) {
            s[j] = x_new[j] - x[j];
            yv[j] = g_new[j] - g[j];
        }

        if ((int)s_list.size() == m_) {
            s_list.pop_front();
            y_list.pop_front();
        }
        s_list.push_back(s);
        y_list.push_back(yv);

        x = std::move(x_new);
    }
    return x;
}

Vec LBFGS::optimize(const Function& f,
                    const Gradient& grad,
                    const Vec& x0) {
    Vec x = x0;
    return optimize(f, grad, x, nullptr);
}

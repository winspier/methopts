#include <iostream>
#include "NewtonOptimizer.h"

struct RosenbrockLagrangian {
    double operator()(const Vec& v) const {
        double x = v[0], y = v[1], l = v[2];
        double f = (1 - x)*(1 - x) + 50 * (y - x * x)*(y - x * x);
        double g = y + x * x;
        return f + l * g;
    }

    Vec grad(const Vec& v) const {
        double x = v[0], y = v[1], l = v[2];
        return {
            -2*(1 - x) - 200 * (y - x*x) * 2 * x + l * 2 * x,
            100 * 2 * (y - x*x) + l,
            y + x * x
        };
    }

    std::vector<Vec> hess(const Vec& v) const {
        double x = v[0], y = v[1], l = v[2];
        return {
                {2 + 400 * (3 * x * x - y) + 2 * l, -400 * x, 2 * x},
                {-400 * x,200,1},
                {2 * x,1,0}
        };
    }
};

int main() {
    RosenbrockLagrangian lagr;
    NewtonOptimizer opt(1e-8, 100);

    std::vector<Vec> starts = {
        {-1.0, 1.0, 0.0},
        {2.0, 4.0, 0.0},
        {0.5, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };

    for (const auto& x0 : starts) {
        Vec result = opt.optimize(
            lagr,
            [&](const Vec& v) { return lagr.grad(v); },
            [&](const Vec& v) { return lagr.hess(v); },
            x0
        );
        std::cout << "Start: ";
        for (double val : x0) std::cout << val << " ";
        std::cout << "\nResult: ";
        for (double val : result) std::cout << val << " ";
        std::cout << "\n----\n";
    }
    return 0;
}

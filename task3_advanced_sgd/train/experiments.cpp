#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include "Optimizers.h"

using Vec = std::vector<double>;
using Function = std::function<double(const Vec&)>;
using Gradient = std::function<Vec(const Vec&)>;

static double rosen(const Vec& x) {
    double a = 1.0, b = 100.0;
    double x0 = x[0], x1 = x[1];
    return (a - x0)*(a - x0) + b*(x1 - x0*x0)*(x1 - x0*x0);
}
static Vec grad_rosen(const Vec& x) {
    double a = 1.0, b = 100.0;
    double x0 = x[0], x1 = x[1];
    double dfdx = -2*(a - x0) - 4*b*x0*(x1 - x0*x0);
    double dfdy =  2*b*(x1 - x0*x0);
    return Vec{ dfdx, dfdy };
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_csv_path>\n";
        return 1;
    }
    std::string out_fname = argv[1];

    Vec learning_rates = { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
    int max_iters = 10000;
    Vec x0 = { -1.2, 1.0 };

    std::ofstream fout(out_fname);
    if (!fout) {
        std::cerr << "Error opening file: " << out_fname << "\n";
        return 1;
    }
    fout << "method,lr,iter,fval\n";

    for (double lr : learning_rates) {
        {
            GradientDescent gd(lr, max_iters);
            OptimizerResult r = gd.optimize(rosen, grad_rosen, x0);
            for (size_t i = 0; i < r.history.size(); ++i) {
                fout << "GD," << lr << "," << i << "," << std::setprecision(12)
                     << r.history[i] << "\n";
            }
            std::cout << "GD converged (lr=" << lr << ") to ["
                      << r.x[0] << ", " << r.x[1] << "] with fval="
                      << std::setprecision(12) << r.history.back() << "\n";
        }
        {
            MomentumGD mgd(lr, max_iters, 0.9);
            OptimizerResult r = mgd.optimize(rosen, grad_rosen, x0);
            for (size_t i = 0; i < r.history.size(); ++i) {
                fout << "Momentum," << lr << "," << i << "," << std::setprecision(12)
                     << r.history[i] << "\n";
            }
            std::cout << "Momentum converged (lr=" << lr << ") to ["
                      << r.x[0] << ", " << r.x[1] << "] with fval="
                      << std::setprecision(12) << r.history.back() << "\n";
        }
        {
            AdamOptimizer adam(lr, max_iters, 0.9, 0.999, 1e-8);
            OptimizerResult r = adam.optimize(rosen, grad_rosen, x0);
            for (size_t i = 0; i < r.history.size(); ++i) {
                fout << "Adam," << lr << "," << i << "," << std::setprecision(12)
                     << r.history[i] << "\n";
            }
            std::cout << "Adam converged (lr=" << lr << ") to ["
                      << r.x[0] << ", " << r.x[1] << "] with fval="
                      << std::setprecision(12) << r.history.back() << "\n";
        }
    }

    fout.close();
    std::cout << "Wrote convergence data to " << out_fname << "\n";
    return 0;
}

#include <iostream>
#include <iomanip>
#include "LBFGS.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }
    int N = std::stoi(argv[1]);
    auto f = [&](const Vec& x) {
        double sum = 0;
        for (int i = 0; i+1 < N; i += 2) {
            double t1 = x[i], t2 = x[i+1];
            sum += 100*(t1*t1 - t2)*(t1*t1 - t2)
                 + (t1 - 1)*(t1 - 1);
        }
        return sum;
    };
    auto grad = [&](const Vec& x) {
        Vec g(N, 0.0);
        for (int i = 0; i+1 < N; i += 2) {
            double t1 = x[i], t2 = x[i+1];
            g[i]   = 400*t1*(t1*t1 - t2) + 2*(t1 - 1);
            g[i+1] = -200*(t1*t1 - t2);
        }
        return g;
    };

    int m = 10, max_iters = 30, tol = 1e-6;
    LBFGS opt(m, max_iters, tol);
    Vec x(N, 0.0);

    std::cout << "iter,loss,grad_norm\n";
    opt.optimize(f, grad, x,
        [&](int iter, const Vec& x_curr, double loss, double grad_norm){
            std::cout << iter << ","
                      << std::setprecision(12) << loss << ","
                      << std::setprecision(12) << grad_norm << "\n";
        }
    );
    return 0;
}

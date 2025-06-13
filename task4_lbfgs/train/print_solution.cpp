#include <iostream>
#include "LBFGS.h"

int main(int argc, char* argv[]) {
    if (argc != 2) { std::cerr << "Usage: " << argv[0] << " N\n"; return 1; }
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
    LBFGS opt(10, 500, 1e-6);
    Vec x(N, 0.0);
    Vec res = opt.optimize(f, grad, x);
    for (double xi : res) std::cout << xi << " ";
    std::cout << "\n";
    return 0;
}

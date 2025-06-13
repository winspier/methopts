#include "LPModel.h"
#include "Simplex.h"
#include <vector>

Vec LPModel::solveRelaxation() const {
    int m = A.size();
    std::vector<std::vector<double>> a_ineq;
    std::vector<double> b_ineq;

    for (int i = 0; i < m; ++i) {
        const Vec& row = A[i];
        double bi = b[i];
        if (isEq[i] == '=') {
            a_ineq.push_back(row);
            b_ineq.push_back(bi);
            Vec neg_row = row;
            for (double& v : neg_row) v = -v;
            a_ineq.push_back(neg_row);
            b_ineq.push_back(-bi);
        } else if (isEq[i] == '>=') {
            Vec neg_row = row;
            for (double& v : neg_row) v = -v;
            a_ineq.push_back(neg_row);
            b_ineq.push_back(-bi);
        } else {
            a_ineq.push_back(row);
            b_ineq.push_back(bi);
        }
    }

    Vec c_max(n);
    for (int j = 0; j < n; ++j) c_max[j] = -c[j];

    Simplex solver(a_ineq, b_ineq, c_max);
    std::vector<double> sol;
    solver.solve(sol);

    return sol;
}

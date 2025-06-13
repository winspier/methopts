#pragma once
#include "common/Types.h"
#include <vector>


struct LPModel {
    int n;
    Vec c;
    std::vector<Vec> A;
    Vec b;
    std::vector<char> isEq;

    LPModel(int n_) : n(n_), c(n_, 0) {}

    void addConstraint(const Vec& a, char op, double bi) {
        A.push_back(a);
        isEq.push_back(op);
        b.push_back(bi);
    }

    Vec solveRelaxation() const;
};

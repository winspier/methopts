#pragma once
#include <vector>
#include <Eigen/Dense>

class Simplex {
public:
    Simplex(const std::vector<std::vector<double>>& a,
            const std::vector<double>& b,
            const std::vector<double>& c);

    double solve(std::vector<double>& solution);

private:
    void pivot(int row, int col);

    int m_, n_;
    Eigen::MatrixXd A_;
    std::vector<int> basic_, non_basic_;
};

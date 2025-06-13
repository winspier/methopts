#include "Simplex.h"
#include <Eigen/Dense>
#include <limits>
#include <iostream>
#include <algorithm>
#include <cmath>

static constexpr double EPS = 1e-9;
static constexpr double INF = 1e18;

Simplex::Simplex(const std::vector<std::vector<double>>& a,
                 const std::vector<double>& b,
                 const std::vector<double>& c)
    : m_(a.size()), n_(a[0].size())
{
    A_.setZero(m_ + 1, n_ + m_ + 1);

    for (int i = 0; i < m_; ++i) {
        for (int j = 0; j < n_; ++j) {
            A_(i, j) = a[i][j];
        }
        A_(i, n_ + i) = 1.0;
        A_(i, n_ + m_) = b[i];
    }

    for (int j = 0; j < n_; ++j) {
        A_(m_, j) = -c[j];
    }

    basic_.resize(m_);
    non_basic_.resize(n_);
    for (int i = 0; i < m_; ++i) basic_[i] = n_ + i;
    for (int j = 0; j < n_; ++j) non_basic_[j] = j;
}

void Simplex::pivot(int row, int col) {
    double pv = A_(row, col);
    A_.row(row) /= pv;

    for (int i = 0; i <= m_; ++i) {
        if (i == row) continue;
        double factor = A_(i, col);
        if (std::abs(factor) < EPS) continue;
        A_.row(i) -= factor * A_.row(row);
    }

    auto it = std::find(non_basic_.begin(), non_basic_.end(), col);
    int idx = std::distance(non_basic_.begin(), it);
    std::swap(basic_[row], non_basic_[idx]);
}

double Simplex::solve(std::vector<double>& solution) {
    while (true) {
        int entering = -1;
        for (int j = 0; j < (int)non_basic_.size(); ++j) {
            int col = non_basic_[j];
            if (A_(m_, col) < -EPS) {
                entering = col;
                break;
            }
        }
        if (entering < 0) break;

        int leaving = -1;
        double best_ratio = INF;
        for (int i = 0; i < m_; ++i) {
            double a_ij = A_(i, entering);
            if (a_ij > EPS) {
                double ratio = A_(i, n_ + m_) / a_ij;
                if (ratio + EPS < best_ratio) {
                    best_ratio = ratio;
                    leaving = i;
                }
            }
        }
        if (leaving < 0) {
            std::cerr << "Unbounded LP\n";
            return std::numeric_limits<double>::infinity();
        }

        pivot(leaving, entering);
    }

    solution.assign(n_, 0.0);
    for (int i = 0; i < m_; ++i) {
        if (basic_[i] < n_) {
            solution[basic_[i]] = A_(i, n_ + m_);
        }
    }
    return A_(m_, n_ + m_);
}

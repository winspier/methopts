#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using Vec = std::vector<double>;

namespace common {

    inline Vec add(const Vec& a, const Vec& b) {
        Vec res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
        return res;
    }

    inline Vec sub(const Vec& a, const Vec& b) {
        Vec res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] - b[i];
        return res;
    }

    inline Vec scalar_mul(const Vec& a, double c) {
        Vec res(a.size());
        for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] * c;
        return res;
    }

    inline double dot(const Vec& a, const Vec& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }

    inline double norm2(const Vec& a) {
        return std::sqrt(dot(a, a));
    }

} // namespace common

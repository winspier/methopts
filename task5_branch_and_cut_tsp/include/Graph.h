#pragma once
#include <vector>

struct Graph {
    int N;
    std::vector<std::vector<double>> cost;

    explicit Graph(int N_);

    void setCost(int i, int j, double c);
};

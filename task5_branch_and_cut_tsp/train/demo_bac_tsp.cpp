#include <iostream>
#include <random>
#include <iomanip>
#include <nlohmann/json.hpp>
#include "Graph.h"
#include "BranchAndCutSolver.h"

int main(int argc, char** argv) {
    int N = 5;
    unsigned seed = 113;
    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) seed = std::atoi(argv[2]);

    Graph G(N);
    std::mt19937 gen(seed);
    std::uniform_real_distribution dist(1.0, 10.0);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double c = dist(gen);
            G.setCost(i, j, c);
        }
    }

    BranchAndCutSolver solver(G);
    TSPSolution sol = solver.solve();

    nlohmann::json js;
    js["N"] = N;
    js["seed"] = seed;
    js["cost"] = nlohmann::json::array();
    for (int i = 0; i < N; ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < N; ++j) {
            row.push_back(G.cost[i][j]);
        }
        js["cost"].push_back(row);
    }
    js["tour"] = sol.tour;
    js["length"] = sol.length;

    std::cout << std::setw(2) << js << std::endl;
    return 0;
}

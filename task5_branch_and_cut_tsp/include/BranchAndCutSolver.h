#pragma once
#include "Graph.h"
#include "LPModel.h"
#include "common/Types.h"
#include <vector>
#include <set>

struct TSPSolution {
    double length;
    std::vector<int> tour;
};

class BranchAndCutSolver {
public:
    BranchAndCutSolver(const Graph& G, int maxNodes = 1000);

    TSPSolution solve();

 private:
    const Graph& G;
    int maxNodes_;

    struct Node {
        std::vector<std::pair<int,int>> fixedEdges;
        std::vector<std::pair<int,int>> forbidden;
    };

    void solveNode(const Node& node);

    LPModel buildLP(const Node& node);

    std::vector<std::set<int>> findSubtours(const Vec& x) const;

    TSPSolution best_;
};

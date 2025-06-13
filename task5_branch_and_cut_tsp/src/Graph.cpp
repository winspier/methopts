#include "Graph.h"

#include <limits>
#include <vector>


Graph::Graph(int N_): N(N_), cost(N_, std::vector(N_, std::numeric_limits<double>::infinity()))
{
    for (int i = 0; i < N; ++i) cost[i][i] = 0;
}

void Graph::setCost(int i, int j, double c)
{
    cost[i][j] = cost[j][i] = c;
}
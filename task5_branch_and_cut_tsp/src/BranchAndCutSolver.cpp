#include "BranchAndCutSolver.h"
#include "LPModel.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <limits>
#include <stdexcept>

static int varIndex(const int i, const int j, const int N) {
    return i * N + j - ((i + 2) * (i + 1)) / 2;
}

BranchAndCutSolver::BranchAndCutSolver(const Graph& G_, const int maxNodes)
    : G(G_), maxNodes_(maxNodes)
{
    best_.length = std::numeric_limits<double>::infinity();
}

TSPSolution BranchAndCutSolver::solve() {
    if (G.N <= 10) {
        std::vector<int> perm(G.N);
        for (int i = 0; i < G.N; ++i) perm[i] = i;
        TSPSolution bestBF;
        bestBF.length = std::numeric_limits<double>::infinity();
        do {
            double len = 0;
            for (int i = 0; i < G.N; ++i) {
                int u = perm[i];
                int v = perm[(i+1)%G.N];
                len += G.cost[u][v];
            }
            if (len < bestBF.length) {
                bestBF.length = len;
                bestBF.tour = perm;
            }
        } while (std::ranges::next_permutation(perm).found);
        return bestBF;
    }

    solveNode(Node{});
    return best_;
}


void BranchAndCutSolver::solveNode(const Node& node) {
    if (maxNodes_-- <= 0) return;

    const int N = G.N;
    const int numVars = N * (N - 1) / 2;

    LPModel lp(numVars);

    {
        int idx = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j, ++idx) {
                lp.c[idx] = G.cost[i][j];
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        Vec row(numVars, 0.0);
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            int k = (i < j ? varIndex(i, j, N) : varIndex(j, i, N));
            row[k] = 1.0;
        }
        lp.addConstraint(row, '=', 2.0);
    }
    for (auto [i, j] : node.fixedEdges) {
        Vec row(numVars, 0.0);
        int k = (i < j ? varIndex(i, j, N) : varIndex(j, i, N));
        row[k] = 1.0;
        lp.addConstraint(row, '=', 1.0);
    }
    for (auto [i, j] : node.forbidden) {
        Vec row(numVars, 0.0);
        int k = (i < j ? varIndex(i, j, N) : varIndex(j, i, N));
        row[k] = 1.0;
        lp.addConstraint(row, '=', 0.0);
    }

    Vec x;
    double lpObj = 0.0;
    while (true) {
        x = lp.solveRelaxation();
        if (x.size() != static_cast<size_t>(numVars)) {
            return;
        }
        lpObj = 0.0;
        for (int k = 0; k < numVars; ++k) {
            lpObj += lp.c[k] * x[k];
        }
        if (lpObj >= best_.length - 1e-9) {
            return;
        }

        bool integral = true;
        for (double xi : x) {
            if (std::abs(xi - std::round(xi)) > 1e-6) {
                integral = false;
                break;
            }
        }
        auto tours = findSubtours(x);
        if (integral) {
            if (tours.size() == 1) {
                double len = 0;
                int idx = 0;
                std::vector<std::vector<int>> adj(G.N);
                for (int i = 0; i < G.N; ++i)
                    for (int j = i+1; j < G.N; ++j, ++idx) {
                        if (x[idx] > 0.5) {
                            len += G.cost[i][j];
                            adj[i].push_back(j);
                            adj[j].push_back(i);
                        }
                    }

                for (int i = 0; i < G.N; ++i)
                    if (adj[i].size() != 2)
                        return;

                if (len < best_.length) {
                    best_.length = len;

                    std::vector<int> path;
                    std::vector visited(G.N, false);

                    int current = 0;
                    int prev = -1;

                    for (int step = 0; step < G.N; ++step) {
                        path.push_back(current);
                        visited[current] = true;

                        int next = -1;
                        for (int v : adj[current]) {
                            if (v != prev) {
                                next = v;
                                break;
                            }
                        }

                        prev = current;
                        current = next;

                        if (current == -1) return;
                    }

                    if (current != path[0]) return;

                    best_.tour = path;
                }
                return;
            }

            for (auto &S : tours) {
                Vec row(numVars, 0.0);
                for (int i : S) {
                    for (int j : S) {
                        if (i < j) {
                            int k = varIndex(i, j, N);
                            row[k] = 1.0;
                        }
                    }
                }
                lp.addConstraint(row, '<', static_cast<double>(S.size() - 1));
            }
        } else {
            if (tours.size() > 1) {
                bool addedCut = false;
                for (auto &S : tours) {
                    if ((int)S.size() < N) {
                        Vec row(numVars, 0.0);
                        for (int i : S) {
                            for (int j : S) {
                                if (i < j) {
                                    int k = varIndex(i, j, N);
                                    row[k] = 1.0;
                                }
                            }
                        }
                        lp.addConstraint(row, '<', static_cast<double>(S.size() - 1));
                        addedCut = true;
                    }
                }
                if (addedCut) {
                    continue;
                }
            }
            int frac_idx = -1;
            double min_dist = 1.0;
            for (int k = 0; k < numVars; ++k) {
                double xi = x[k];
                if (xi > 1e-6 && xi < 1 - 1e-6) {
                    double dist = std::abs(xi - 0.5);
                    if (dist < min_dist) {
                        min_dist = dist;
                        frac_idx = k;
                    }
                }
            }
            if (frac_idx < 0) {
                return;
            }
            int cnt = 0;
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j, ++cnt) {
                    if (cnt == frac_idx) {
                        Node left = node;
                        left.forbidden.emplace_back(i, j);
                        Node right = node;
                        right.fixedEdges.emplace_back(i, j);
                        solveNode(left);
                        solveNode(right);
                        return;
                    }
                }
            }
            return;
        }
    }
}

std::vector<std::set<int>> BranchAndCutSolver::findSubtours(const Vec& x) const
{
    int N = G.N;
    std::vector<std::vector<int>> adj(N);
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j, ++idx) {
            if (x[idx] > 0.5) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    std::vector used(N,false);
    std::vector<std::set<int>> comps;
    for (int i = 0; i < N; ++i) {
        if (!used[i]) {
            std::queue<int> q;
            q.push(i);
            used[i] = true;
            std::set<int> comp;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                comp.insert(u);
                for (int v : adj[u]) {
                    if (!used[v]) {
                        used[v] = true;
                        q.push(v);
                    }
                }
            }
            comps.push_back(comp);
        }
    }
    return comps;
}

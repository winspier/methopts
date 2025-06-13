#include <gtest/gtest.h>
#include "Graph.h"
#include "BranchAndCutSolver.h"

TEST(BranchAndCutTest, Square4) {
    Graph G(4);
    G.setCost(0,1,1);
    G.setCost(1,2,1);
    G.setCost(2,3,1);
    G.setCost(3,0,1);
    G.setCost(0,2,2);
    G.setCost(1,3,2);

    BranchAndCutSolver solver(G);
    auto sol = solver.solve();

    EXPECT_NEAR(sol.length, 4.0, 1e-6);

    ASSERT_EQ(sol.tour.size(), 4u);
    std::set vs(sol.tour.begin(), sol.tour.end());
    EXPECT_EQ(vs.size(), 4u);
}

TEST(BranchAndCutTest, Triangle3) {
    Graph G(3);
    G.setCost(0,1,5);
    G.setCost(1,2,7);
    G.setCost(0,2,9);

    BranchAndCutSolver solver(G);
    auto sol = solver.solve();

    EXPECT_NEAR(sol.length, 5+7+9, 1e-6);

    ASSERT_EQ(sol.tour.size(), 3u);
    std::set vs(sol.tour.begin(), sol.tour.end());
    EXPECT_EQ(vs.size(), 3u);
}

TEST(BranchAndCutTest, Simple4_CheckTourOrder) {
    Graph g(4);
    g.cost = {
        {0, 1, 2, 1},
        {1, 0, 2, 3},
        {2, 2, 0, 2},
        {1, 3, 2, 0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    ASSERT_NEAR(sol.length, 6.0, 1e-6);
    ASSERT_EQ(sol.tour.size(), 4u);

    std::vector seen(4, false);
    for (int v : sol.tour) {
        ASSERT_GE(v, 0);
        ASSERT_LT(v, 4);
        seen[v] = true;
    }

    for (bool s : seen) {
        ASSERT_TRUE(s);
    }

    for (size_t i = 0; i < sol.tour.size(); ++i) {
        int u = sol.tour[i];
        int v = sol.tour[(i+1) % sol.tour.size()];
        ASSERT_NE(g.cost[u][v], 0);
    }
}

TEST(BranchAndCutTest, ReconstructTourCorrectly) {
    Graph g(5);
    g.cost = {
        {0, 2, 9, 10, 7},
        {2, 0, 6, 4, 3},
        {9, 6, 0, 8, 5},
        {10,4, 8, 0, 6},
        {7, 3, 5, 6, 0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    ASSERT_GT(sol.length, 0.0);
    ASSERT_EQ(sol.tour.size(), 5u);

    std::vector seen(5, false);
    for (int v : sol.tour) {
        ASSERT_GE(v, 0);
        ASSERT_LT(v, 5);
        ASSERT_FALSE(seen[v]);
        seen[v] = true;
    }

    double sum = 0.0;
    for (size_t i = 0; i < sol.tour.size(); ++i) {
        int u = sol.tour[i];
        int v = sol.tour[(i+1)%sol.tour.size()];
        sum += g.cost[u][v];
    }

    ASSERT_NEAR(sum, sol.length, 1e-6);
}

TEST(BranchAndCutTest, TourMustBeReturnedWithLength) {
    Graph g(4);
    g.cost = {
        {0, 1, 2, 1},
        {1, 0, 2, 3},
        {2, 2, 0, 2},
        {1, 3, 2, 0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    ASSERT_NEAR(sol.length, 6.0, 1e-6);

    ASSERT_FALSE(sol.tour.empty()) << "Tour was not reconstructed";

    double sum = 0;
    for (size_t i = 0; i < sol.tour.size(); ++i) {
        int u = sol.tour[i];
        int v = sol.tour[(i+1) % sol.tour.size()];
        sum += g.cost[u][v];
    }

    ASSERT_NEAR(sum, sol.length, 1e-6) << "Tour length does not match edge sum";
}

TEST(BranchAndCutTest, TourLengthMustMatchSum) {
    Graph g(4);
    g.cost = {
        {0, 1, 2, 1},
        {1, 0, 2, 3},
        {2, 2, 0, 2},
        {1, 3, 2, 0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    EXPECT_NEAR(sol.length, 6.0, 1e-6) << "Expected length 6";

    double sum = 0.0;
    for (size_t i = 0; i < sol.tour.size(); ++i) {
        int u = sol.tour[i];
        int v = sol.tour[(i + 1) % sol.tour.size()];
        sum += g.cost[u][v];
    }

    EXPECT_NEAR(sum, sol.length, 1e-6) << "Tour not reconstructed or incorrect";
}

TEST(BranchAndCutTest, TourSizeIsN) {
    Graph g(5);
    g.cost = {
        {0,1,2,3,4},
        {1,0,5,6,7},
        {2,5,0,8,9},
        {3,6,8,0,10},
        {4,7,9,10,0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    ASSERT_EQ(sol.tour.size(), g.N) << "Tour must contain all vertices";
}

TEST(BranchAndCutTest, TourIsPermutation) {
    Graph g(4);
    g.cost = {
        {0,1,2,3},
        {1,0,4,5},
        {2,4,0,6},
        {3,5,6,0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    std::vector seen(g.N, false);
    for (int v : sol.tour) {
        ASSERT_GE(v, 0);
        ASSERT_LT(v, g.N);
        ASSERT_FALSE(seen[v]) << "Vertex repeated in tour";
        seen[v] = true;
    }
}

TEST(BranchAndCutTest, TourLengthMatchesCost) {
    Graph g(4);
    g.cost = {
        {0,1,2,3},
        {1,0,4,5},
        {2,4,0,6},
        {3,5,6,0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    double total = 0.0;
    for (size_t i = 0; i < sol.tour.size(); ++i) {
        int u = sol.tour[i];
        int v = sol.tour[(i + 1) % sol.tour.size()];
        total += g.cost[u][v];
    }

    ASSERT_NEAR(total, sol.length, 1e-6) << "Tour length must match edge costs";
}

TEST(BranchAndCutTest, TourMustBeCycle) {
    Graph g(3);
    g.cost = {
        {0, 2, 3},
        {2, 0, 1},
        {3, 1, 0}
    };

    BranchAndCutSolver solver(g);
    TSPSolution sol = solver.solve();

    ASSERT_FALSE(sol.tour.empty()) << "Tour must not be empty";

    int start = sol.tour.front();
    int end   = sol.tour.back();
    double lastEdge = g.cost[end][start];

    ASSERT_LT(lastEdge, 1e9) << "Tour must be closed (end → start)";
}

TEST(BranchAndCutTest, RecoversHamiltonianCycleCorrectly) {
    int N = 6;
    Graph G(N);
    G.cost = std::vector(N, std::vector(N, 100.0));
    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        G.cost[i][j] = 1.0;
        G.cost[j][i] = 1.0;
    }

    BranchAndCutSolver solver(G, 100);
    TSPSolution sol = solver.solve();

    EXPECT_NEAR(sol.length, N, 1e-6);
    ASSERT_EQ(sol.tour.size(), N);
    std::vector seen(N, false);
    for (int v : sol.tour) {
        ASSERT_GE(v, 0);
        ASSERT_LT(v, N);
        EXPECT_FALSE(seen[v]);
        seen[v] = true;
    }
}

TEST(BranchAndCutTest, TourIsBuiltWhenIntegerSolutionFound) {
    int N = 9;
    Graph G(N);
    G.cost = std::vector(N, std::vector(N, 100.0));

    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        G.cost[i][j] = 1.0;
        G.cost[j][i] = 1.0;
    }

    BranchAndCutSolver solver(G, 1000);
    TSPSolution solution = solver.solve();

    EXPECT_DOUBLE_EQ(solution.length, N * 1.0);

    ASSERT_EQ(solution.tour.size(), N);

    double actualLength = 0.0;
    std::vector<bool> visited(N, false);

    for (int i = 0; i < N; ++i) {
        int u = solution.tour[i];
        int v = solution.tour[(i + 1) % N];

        EXPECT_FALSE(visited[u]) << "Vertex " << u << " is repeated";
        visited[u] = true;

        actualLength += G.cost[u][v];
    }

    EXPECT_DOUBLE_EQ(actualLength, N * 1.0);
}

TEST(BranchAndCutTest, TourIsBuiltWhenIntegerSolutionFound2) {
    int N = 5;
    Graph G(N);
    G.cost = std::vector(N, std::vector(N, 10.0));

    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        G.cost[i][j] = 1.0;
        G.cost[j][i] = 1.0;
    }

    BranchAndCutSolver solver(G, 1000);
    TSPSolution solution = solver.solve();

    EXPECT_DOUBLE_EQ(solution.length, N * 1.0);

    ASSERT_EQ(solution.tour.size(), N);
    std::vector visited(N, false);
    double actualLength = 0.0;

    for (int i = 0; i < N; ++i) {
        int u = solution.tour[i];
        int v = solution.tour[(i + 1) % N];
        EXPECT_FALSE(visited[u]) << "Vertex " << u << " repeated";
        visited[u] = true;
        actualLength += G.cost[u][v];
    }

    EXPECT_DOUBLE_EQ(actualLength, N * 1.0);
}

TEST(BranchAndCutTest, TourIsBuiltWhenIntegerSolutionFound3) {
    int N = 9;
    Graph G(N);
    G.cost = std::vector(N, std::vector<double>(N, 100.0));

    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        G.cost[i][j] = 1.0;
        G.cost[j][i] = 1.0;
    }

    BranchAndCutSolver solver(G, 1000);
    TSPSolution solution = solver.solve();

    EXPECT_NEAR(solution.length, N, 1e-6);

    ASSERT_EQ(solution.tour.size(), N);

    std::vector<int> count(N, 0);
    for (int v : solution.tour) {
        ASSERT_GE(v, 0);
        ASSERT_LT(v, N);
        count[v]++;
    }
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(count[i], 1) << "Vertex " << i << " appears " << count[i] << " times";
    }

    double actualLength = 0.0;
    for (int i = 0; i < N; ++i) {
        int u = solution.tour[i];
        int v = solution.tour[(i + 1) % N];
        actualLength += G.cost[u][v];
    }
    EXPECT_NEAR(actualLength, N, 1e-6);
}

TEST(BranchAndCutTest, TourIsBuiltCorrectly) {
    int N = 9;
    Graph G(N);
    G.cost = std::vector(N, std::vector(N, 100.0));

    for (int i = 0; i < N; ++i) {
        int j = (i + 1) % N;
        G.cost[i][j] = 1.0;
        G.cost[j][i] = 1.0;
    }

    BranchAndCutSolver solver(G, 10000);
    TSPSolution solution = solver.solve();

    EXPECT_NEAR(solution.length, N * 1.0, 1e-6);

    ASSERT_EQ(solution.tour.size(), N);

    std::vector visited(N, false);
    for (int i = 0; i < N; ++i) {
        int v = solution.tour[i];
        EXPECT_FALSE(visited[v]) << "Vertex " << v << " repeated";
        visited[v] = true;
    }

    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(visited[i]) << "Vertex " << i << " missing";
    }

    double actualLength = 0.0;
    for (int i = 0; i < N; ++i) {
        int u = solution.tour[i];
        int v = solution.tour[(i + 1) % N];
        actualLength += G.cost[u][v];
    }
    EXPECT_NEAR(actualLength, N * 1.0, 1e-6);
}
#!/usr/bin/env python3
import sys
import json
import math
import matplotlib.pyplot as plt

def read_solution(json_path):
    if json_path == "-":
        data = json.load(sys.stdin)
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
    N = data["N"]
    cost = data["cost"]
    tour = data["tour"]
    length = data.get("length", None)
    return N, cost, tour, length


def circle_positions(N, radius=1.0, center=(0,0)):
    cx, cy = center
    angles = [2*math.pi*i/N for i in range(N)]
    pos = [(cx + radius*math.cos(a), cy + radius*math.sin(a)) for a in angles]
    return pos

def plot_tsp(N, cost, tour, length=None, out_path=None):
    pos = circle_positions(N, radius=1.0)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(N):
        xi, yi = pos[i]
        for j in range(i+1, N):
            xj, yj = pos[j]
            ax.plot([xi, xj], [yi, yj], color='gray', linewidth=0.5, zorder=1)

    for i, (x, y) in enumerate(pos):
        ax.scatter(x, y, s=100, color='white', edgecolor='black', zorder=2)
        ax.text(x, y, str(i), ha='center', va='center', zorder=3)

    tour_edges = []
    for idx in range(len(tour)):
        u = tour[idx]
        v = tour[(idx+1) % len(tour)]
        tour_edges.append((u, v))

    for u, v in tour_edges:
        xu, yu = pos[u]
        xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color='red', linewidth=2.0, zorder=4)

    if length is not None:
        ax.set_title(f"TSP tour, length = {length:.4f}")

    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: plot_tsp.py solution.json [output.png]")
        sys.exit(1)
    json_path = sys.argv[1]
    out_path = None
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    N, cost, tour, length = read_solution(json_path)
    plot_tsp(N, cost, tour, length, out_path)

if __name__ == "__main__":
    main()

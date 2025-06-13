#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 6:
    sys.exit("Usage: plot_lagrangian.py x_min x_max y_min y_max lambda")

x_min, x_max = float(sys.argv[1]), float(sys.argv[2])
y_min, y_max = float(sys.argv[3]), float(sys.argv[4])
lambda_val  = float(sys.argv[5])

def L(x, y, l):
    return (1 - x)**2 + 50*(y - x**2)**2 + l*(y + x**2)

x_vals = np.linspace(x_min, x_max, 300)
y_vals = np.linspace(y_min, y_max, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = L(X, Y, lambda_val)

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, Z, levels=100, cmap='viridis')
plt.colorbar(cp)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Lagrangian landscape (λ = {lambda_val})")

constraint_x = x_vals
constraint_y = -constraint_x**2
plt.plot(constraint_x, constraint_y, 'r-', linewidth=2, label='y = -x²')
plt.legend()

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"lagrangian_lambda_{lambda_val}.png")
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved figure to {out_path}")

plt.show()

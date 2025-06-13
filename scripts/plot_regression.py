import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

if len(sys.argv) < 4:
    print("Usage: plot_regression.py data.tsv beta0 beta1")
    sys.exit(1)

data_file = sys.argv[1]
beta0 = float(sys.argv[2])
beta1 = float(sys.argv[3])

x1_list, x2_list, y_list = [], [], []
with open(data_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader, None)
    for row in reader:
        if len(row) < 3:
            continue
        x1, x2, y = map(float, row[:3])
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y)

os.makedirs("output", exist_ok=True)
base_name = os.path.splitext(os.path.basename(data_file))[0]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x1_list, x2_list, y_list, color='blue', label='Observed y')
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("y")
ax1.set_title("Before regression")
ax1.legend()
path1 = os.path.join("output", f"{base_name}_before_regression.png")
plt.savefig(path1)
print(f"Saved: {path1}")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x1_list, x2_list, y_list, color='blue', label='Observed y')

x1_range = np.linspace(min(x1_list), max(x1_list), 30)
x2_range = np.linspace(min(x2_list), max(x2_list), 30)
X1, X2 = np.meshgrid(x1_range, x2_range)
Y_pred = beta0 * X1 + beta1 * X2
ax2.plot_surface(X1, X2, Y_pred, alpha=0.5, color='green')

proxy_data = Line2D([0], [0], linestyle="none", marker='o', color='blue')
proxy_plane = Line2D([0], [0], linestyle="none", marker='s', color='green')
ax2.legend([proxy_data, proxy_plane], ['Observed y', 'Regression plane'])

ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("y")
ax2.set_title("After regression")
path2 = os.path.join("output", f"{base_name}_after_regression.png")
plt.savefig(path2)
print(f"Saved: {path2}")

plt.show()

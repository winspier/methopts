#!/usr/bin/env python3
import sys, csv
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: plot_lbfgs.py history.csv")
    sys.exit(1)

history_file = sys.argv[1]
iters, losses, grad_norms = [], [], []
with open(history_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        it, loss, gn = row
        iters.append(int(it))
        losses.append(float(loss))
        grad_norms.append(float(gn))

plt.figure()
plt.plot(iters, losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('L-BFGS Loss Convergence')
plt.legend()
plt.savefig('output/lbfgs_loss.png')
print("Saved: output/lbfgs_loss.png")

plt.figure()
plt.plot(iters, grad_norms, label='||grad||')
plt.xlabel('Iteration')
plt.ylabel('Gradient norm')
plt.title('L-BFGS Gradient Norm')
plt.legend()
plt.savefig('output/lbfgs_grad_norm.png')
print("Saved: output/lbfgs_grad_norm.png")

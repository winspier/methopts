#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

MAX_FVAL = 1e10

def plot_convergence(df, csv_path, output_path=None):
    methods = df['method'].unique()
    lrs = sorted(df['lr'].unique())

    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "convergence.png")

    plt.figure(figsize=(8, 6))
    for method in methods:
        for lr in lrs:
            sub = df[(df['method'] == method) & (df['lr'] == lr)]
            if sub.empty:
                continue
            if sub['fval'].max() > MAX_FVAL:
                print(f"Warning: Possible divergence in {method}, lr={lr}")
            sub = sub[sub['fval'] < MAX_FVAL]
            if sub.empty:
                continue
            plt.plot(sub['iter'], sub['fval'], label=f"{method}, lr={lr}")

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.title('Convergence on Rosenbrock')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True, which='both', ls='--', lw=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")
    plt.close()


def plot_per_method(df, csv_path):
    methods = df['method'].unique()
    lrs = sorted(df['lr'].unique())

    for method in methods:
        plt.figure(figsize=(8, 6))
        for lr in lrs:
            sub = df[(df['method'] == method) & (df['lr'] == lr)]
            if sub.empty:
                continue
            if sub['fval'].max() > MAX_FVAL:
                print(f"Warning: Possible divergence in {method}, lr={lr}")
            sub = sub[sub['fval'] < MAX_FVAL]
            if sub.empty:
                continue
            plt.plot(sub['iter'], sub['fval'], label=f"lr={lr}")

        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('f(x)')
        plt.title(f'Convergence of {method} on Rosenbrock')
        plt.legend(fontsize='small')
        plt.grid(True, which='both', ls='--', lw=0.5)

        out_file = os.path.join(os.path.dirname(csv_path), f"convergence_{method}.png")
        plt.tight_layout()
        plt.savefig(out_file, bbox_inches='tight')
        print(f"Saved per-method plot to {out_file}")
        plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_convergence.py convergence.csv [output.png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"❌ Error: File not found: {csv_path}")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("❌ Error: CSV is empty or malformed.")
        sys.exit(1)

    required_columns = {'method', 'lr', 'iter', 'fval'}
    if not required_columns.issubset(df.columns):
        print(f"❌ Error: CSV must contain columns: {', '.join(required_columns)}")
        sys.exit(1)

    plot_convergence(df, csv_path, output_path)
    plot_per_method(df, csv_path)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import subprocess
import sys
import os

def die(msg):
    sys.exit(f"Error: {msg}")

if len(sys.argv) != 4:
    die("Usage: train_and_plot_regression.py <data_file.tsv> <build_dir> <output_dir>")

data_file, build_dir, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
print(f"Using build_dir = {build_dir}")
print(f"Using output_dir = {output_dir}")

if not os.path.isdir(build_dir):
    die(f"build_dir does not exist: {build_dir}")
if not os.path.isdir(output_dir):
    die(f"output_dir does not exist: {output_dir}")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", data_file)
if not os.path.isfile(data_path):
    die(f"Data file not found: {data_path}")

train_exec = os.path.join(build_dir, "train_linear_regression")
if not os.path.isfile(train_exec) or not os.access(train_exec, os.X_OK):
    die(f"Cannot execute training binary: {train_exec}")

print(f"Running: {train_exec} {data_path} {output_dir}")
try:
    subprocess.check_call([train_exec, data_path, output_dir])
except subprocess.CalledProcessError as e:
    die(f"Training failed with exit code {e.returncode}")

beta_file = os.path.join(output_dir, "beta.txt")
if not os.path.isfile(beta_file):
    die(f"beta.txt not found in output_dir: {beta_file}")

with open(beta_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
if len(lines) < 2:
    die(f"beta.txt should contain at least two lines (beta0 and beta1), got {len(lines)}")

beta0, beta1 = lines[0], lines[1]
print(f"Read beta0={beta0}, beta1={beta1}")

plot_script = os.path.join(project_root, "scripts", "plot_regression.py")
if not os.path.isfile(plot_script):
    die(f"plot_regression.py not found: {plot_script}")

print(f"Running: {sys.executable} {plot_script} {data_path} {beta0} {beta1}")
try:
    subprocess.check_call([
        sys.executable, plot_script,
        data_path,
        beta0, beta1
    ])
except subprocess.CalledProcessError as e:
    die(f"Plotting failed with exit code {e.returncode}")

print("Done.")

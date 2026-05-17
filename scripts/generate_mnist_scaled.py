from sklearn.datasets import fetch_openml
import numpy as np
import csv
import os
import argparse

parser = argparse.ArgumentParser(description="Generate scaled MNIST CSV dataset.")
parser.add_argument("--samples-per-class", type=int, default=600, 
                    help="Number of samples to draw per class (0-10) (default: 600 for 6,000 total samples)")
parser.add_argument("--output", type=str, default="data/mnist_scaled.csv",
                    help="Output CSV file path (default: data/mnist_scaled.csv)")
args = parser.parse_args()

print("Fetching MNIST dataset from OpenML...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

samples_per_class = args.samples_per_class
print(f"Sampling {samples_per_class} samples per class (Total: {samples_per_class * 10} samples)...")

indices = []
for c in range(10):
    class_indices = np.where(y == c)[0]
    if len(class_indices) < samples_per_class:
        raise ValueError(f"Class {c} only has {len(class_indices)} samples, but {samples_per_class} were requested.")
    idx = class_indices[:samples_per_class]
    indices.extend(idx)

np.random.shuffle(indices)

os.makedirs('data', exist_ok=True)
output_file = args.output

print(f"Saving to {output_file}...")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'p{i}' for i in range(784)])
    for idx, i in enumerate(indices):
        writer.writerow([y[i]] + X[i].tolist())
        if (idx + 1) % 1000 == 0:
            print(f"Written {idx + 1} / {len(indices)} samples...")

print(f"Successfully saved {len(indices)} samples to {output_file}!")

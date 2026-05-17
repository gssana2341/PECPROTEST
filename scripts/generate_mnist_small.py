from sklearn.datasets import fetch_openml
import numpy as np
import csv
import os

print("Fetching MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Sample 100 per class -> 1,000 samples
samples_per_class = 100
indices = []
for c in range(10):
    idx = np.where(y == c)[0][:samples_per_class]
    indices.extend(idx)
np.random.shuffle(indices)

os.makedirs('data', exist_ok=True)
output_file = 'data/mnist_small.csv'

print(f"Saving to {output_file}...")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'p{i}' for i in range(784)])
    for i in indices:
        writer.writerow([y[i]] + X[i].tolist())

print(f"Saved {len(indices)} samples")

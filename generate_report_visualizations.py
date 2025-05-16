import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Read the JSON files
def load_results(filename):
    with open(f'results/{filename}', 'r') as f:
        return json.load(f)

baseline_results = load_results('baseline_results.json')
random_shuffle_results = load_results('random_shuffle_results.json')
label_noise_results = load_results('label_noise_results.json')
input_perturbation_results = load_results('input_perturbation_results.json')

# CIFAR-10 classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. Comparison of test accuracy across epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(baseline_results['test_accs'])+1), baseline_results['test_accs'], 'o-', label='Baseline')
plt.plot(range(1, len(random_shuffle_results['test_accs'])+1), random_shuffle_results['test_accs'], 's-', label='Random Label Shuffle')
plt.plot(range(1, len(label_noise_results['test_accs'])+1), label_noise_results['test_accs'], '^-', label='Label Noise (20%)')
plt.plot(range(1, len(input_perturbation_results['test_accs'])+1), input_perturbation_results['test_accs'], 'D-', label='Input Perturbation')

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Comparison Across Experimental Conditions')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('results/comparison_test_acc.png', dpi=300)
plt.close()

# 2. Per-class accuracy comparison
plt.figure(figsize=(15, 8))
bar_width = 0.2
index = np.arange(len(classes))

# Calculate per-class accuracy for each condition
def get_class_accuracy(results):
    return [100 * results['class_correct'][i] / results['class_total'][i] for i in range(10)]

baseline_class_acc = get_class_accuracy(baseline_results)
random_shuffle_class_acc = get_class_accuracy(random_shuffle_results)
label_noise_class_acc = get_class_accuracy(label_noise_results)
input_perturbation_class_acc = get_class_accuracy(input_perturbation_results)

plt.bar(index - 1.5*bar_width, baseline_class_acc, bar_width, label='Baseline')
plt.bar(index - 0.5*bar_width, random_shuffle_class_acc, bar_width, label='Random Label Shuffle')
plt.bar(index + 0.5*bar_width, label_noise_class_acc, bar_width, label='Label Noise (20%)')
plt.bar(index + 1.5*bar_width, input_perturbation_class_acc, bar_width, label='Input Perturbation')

plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Per-class Accuracy Comparison')
plt.xticks(index, classes, rotation=45)
plt.legend()
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/per_class_comparison.png', dpi=300)
plt.close()

# 3. Training curves for baseline condition
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(baseline_results['train_losses'])+1), baseline_results['train_losses'], 'o-', label='Train Loss')
plt.plot(range(1, len(baseline_results['test_losses'])+1), baseline_results['test_losses'], 's-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(baseline_results['train_accs'])+1), baseline_results['train_accs'], 'o-', label='Train Accuracy')
plt.plot(range(1, len(baseline_results['test_accs'])+1), baseline_results['test_accs'], 's-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('results/baseline_curves.png', dpi=300)
plt.close()

print("All visualization images have been created in the 'results' directory.")

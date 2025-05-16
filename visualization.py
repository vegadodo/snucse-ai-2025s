import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(results, exp_name):
    """Plot training and test curves for a single experiment."""
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{exp_name} - Loss Curves')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(results['train_accs'], label='Train Accuracy')
    plt.plot(results['test_accs'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{exp_name} - Accuracy Curves')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{exp_name}_curves.png')
    plt.close()

def plot_class_accuracy(class_acc, classes, exp_name):
    """Plot per-class accuracy for a single experiment."""
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_acc)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{exp_name} - Per-class Accuracy')
    plt.ylim([0, 100])
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{exp_name}_class_acc.png')
    plt.close()

def plot_comparison(all_results, metric='test_acc', title='Test Accuracy Comparison'):
    """Compare results across all experiments."""
    plt.figure(figsize=(12, 6))

    for exp_name, results in all_results.items():
        if metric == 'test_acc':
            plt.plot(results['test_accs'], label=exp_name)
            ylabel = 'Accuracy (%)'
        elif metric == 'test_loss':
            plt.plot(results['test_losses'], label=exp_name)
            ylabel = 'Loss'

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/comparison_{metric}.png')
    plt.close()

def plot_final_comparison(all_results, classes):
    """Plot final accuracy comparison and per-class accuracy comparison."""
    # Final accuracy comparison
    plt.figure(figsize=(10, 6))
    exp_names = list(all_results.keys())
    final_accs = [all_results[exp]['test_accs'][-1] for exp in exp_names]

    plt.bar(exp_names, final_accs)
    plt.xlabel('Experiment')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.ylim([0, 100])
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/final_accuracy_comparison.png')
    plt.close()

    # Per-class accuracy comparison
    plt.figure(figsize=(15, 8))
    bar_width = 0.2
    index = np.arange(len(classes))

    for i, (exp_name, results) in enumerate(all_results.items()):
        class_accs = []
        for j in range(10):
            if results['class_total'][j] > 0:
                acc = 100 * results['class_correct'][j] / results['class_total'][j]
            else:
                acc = 0
            class_accs.append(acc)

        plt.bar(index + i*bar_width, class_accs, bar_width, label=exp_name)

    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-class Accuracy Comparison')
    plt.xticks(index + bar_width, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/per_class_comparison.png')
    plt.close()

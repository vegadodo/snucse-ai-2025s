import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
from data_utils import get_data_loaders, get_cifar10_classes
from model import BasicCNN, train_epoch, evaluate_model
from visualization import (
    plot_training_curves, plot_class_accuracy,
    plot_comparison, plot_final_comparison
)

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate(exp_name, trainloader, testloader, device, num_epochs=20, lr=0.001):
    """Run a complete training and evaluation process."""
    model = BasicCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Get class names
    classes = get_cifar10_classes()

    # Track results
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0.0

    # Main training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - {exp_name}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, trainloader, device, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate the model
        test_loss, test_acc, class_correct, class_total = evaluate_model(
            model, testloader, device, criterion, classes)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update learning rate
        scheduler.step(test_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{exp_name}_best.pth')
            print("Best model saved!")

    # Final evaluation with best model
    model.load_state_dict(torch.load(f'models/{exp_name}_best.pth'))
    final_loss, final_acc, class_correct, class_total = evaluate_model(
        model, testloader, device, criterion, classes)

    # Prepare results dictionary
    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_loss': final_loss,
        'final_acc': final_acc,
        'class_correct': class_correct,
        'class_total': class_total,
        'best_acc': best_acc
    }

    # Plot and save individual results
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    plot_training_curves(results, exp_name)
    plot_class_accuracy(class_accuracy, classes, exp_name)

    return results

def main():
    # Configuration
    seed = 42
    num_epochs = 20
    batch_size = 128
    lr = 0.001

    # Set random seed
    set_seed(seed)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define experiment types
    experiments = [
        'baseline',
        'random_shuffle',
        'label_noise',
        'input_perturbation'
    ]

    # Run all experiments and collect results
    all_results = {}

    for exp_type in experiments:
        print(f"\n{'='*20} Running {exp_type} experiment {'='*20}")
        trainloader, testloader = get_data_loaders(exp_type, batch_size)

        # Train and evaluate
        results = train_and_evaluate(
            exp_type, trainloader, testloader, device, num_epochs, lr)

        # Store results
        all_results[exp_type] = results

        # Save results to JSON
        with open(f'results/{exp_type}_results.json', 'w') as f:
            # Convert NumPy arrays to lists for JSON serialization
            serializable_results = {
                k: v if not isinstance(v, (np.ndarray, list)) or k in ['class_correct', 'class_total']
                else [float(item) for item in v]
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=4)

    # Plot comparisons
    plot_comparison(all_results, 'test_acc', 'Test Accuracy Comparison')
    plot_comparison(all_results, 'test_loss', 'Test Loss Comparison')
    plot_final_comparison(all_results, get_cifar10_classes())

    print("\nAll experiments completed! Results saved to 'results' directory.")

if __name__ == '__main__':
    main()

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

def get_base_transforms():
    """Return the base transforms for train and test data."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    return transform_train, transform_test

def get_perturbed_transforms():
    """Return transforms with significant image perturbations."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),  # More aggressive cropping
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Add rotation
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Color perturbation
        transforms.GaussianBlur(kernel_size=3),  # Add blur
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Test transforms remain the same as baseline for fair comparison
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    return transform_train, transform_test

class CIFAR10WithRandomLabels(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset with randomly assigned labels."""
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10WithRandomLabels, self).__init__(root, train, transform, target_transform, download)
        # Generate random labels
        self.targets = torch.randint(0, 10, (len(self.targets),)).tolist()

class CIFAR10WithNoisyLabels(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset with noisy labels (some percentage of labels are randomized)."""
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, noise_ratio=0.2):
        super(CIFAR10WithNoisyLabels, self).__init__(root, train, transform, target_transform, download)

        if train:  # Only add noise to training data
            # Convert targets to numpy array for easier manipulation
            targets = np.array(self.targets)

            # Determine which indices to add noise to
            num_samples = len(targets)
            num_noise = int(num_samples * noise_ratio)
            noise_indices = random.sample(range(num_samples), num_noise)

            # Add noise (change labels to a random class different from original)
            for idx in noise_indices:
                original_label = targets[idx]
                possible_labels = list(range(10))
                possible_labels.remove(original_label)
                targets[idx] = random.choice(possible_labels)

            # Set the new noisy targets
            self.targets = targets.tolist()

def get_data_loaders(experiment_type, batch_size=128):
    """Return data loaders for the specified experiment type."""
    if experiment_type == 'baseline':
        # Baseline: Original CIFAR-10
        transform_train, transform_test = get_base_transforms()
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    elif experiment_type == 'random_shuffle':
        # Random Label Shuffle: CIFAR-10 with random labels
        transform_train, transform_test = get_base_transforms()
        trainset = CIFAR10WithRandomLabels(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    elif experiment_type == 'label_noise':
        # Label Noise: CIFAR-10 with 20% noisy labels
        transform_train, transform_test = get_base_transforms()
        trainset = CIFAR10WithNoisyLabels(
            root='./data', train=True, download=True, transform=transform_train, noise_ratio=0.2)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    elif experiment_type == 'input_perturbation':
        # Input Perturbation: CIFAR-10 with perturbed images
        transform_train, transform_test = get_perturbed_transforms()
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader

def get_cifar10_classes():
    """Return the class names for CIFAR-10."""
    return ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

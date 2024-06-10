import torch
import torch.nn.functional as F

def compute_class_counts(device, output, target):
    num_classes = output.size()[1]

    # Compute predictions and detach them from the graph
    _, predicted = torch.max(output, 1)
    predicted = predicted.detach()
    target_labels = target.to(device).detach()

    # Flatten tensors for comparison
    predicted = predicted.view(-1)
    target_labels = target_labels.view(-1)

    # Count correct and wrong predictions for each class
    class_counts = []
    for i in range(num_classes):
        class_mask = (target_labels == i)
        class_i = {
            'correct_counts': (predicted[class_mask] == i).sum().item(),
            'wrong_counts': (predicted[class_mask] != i).sum().item()
        }
        class_counts.append(class_i)
        #self.correct_counts[i] += (predicted[class_mask] == i).sum().item()
        #self.wrong_counts[i] += (predicted[class_mask] != i).sum().item()

    # Print the results
    for i in range(num_classes):
        print(f"Class {i}: Correct predictions: {class_counts[i]['correct_counts']}, Wrong predictions: {class_counts[i]['wrong_counts']}")

    return class_counts


# importing libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Data transformations

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading dataset

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Defining the model

class ResNetObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(ResNetObjectDetector, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 4)
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        return class_logits, bbox_preds

# Initializing model, loss functions, and optimizer

num_classes = len(cifar10_classes)
model = ResNetObjectDetector(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.MSELoss()

# Defining Optimizer and hyperparameters
 
learning_rate = 0.001
weight_decay = 0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# mAP calculation

def calculate_map(pred_probs, true_labels, num_classes):
    average_precisions = []
    for class_idx in range(num_classes):
        true_positive = []
        scores = []
        num_gt = sum([1 for t in true_labels if t == class_idx])
        
        if num_gt == 0:
            continue  
        
        for i, (pred_prob, true_label) in enumerate(zip(pred_probs, true_labels)):
            scores.append(pred_prob[class_idx])
            true_positive.append(1 if true_label == class_idx else 0)

        
        sorted_indices = np.argsort(scores)[::-1]
        true_positive = np.array(true_positive)[sorted_indices]
        cumulative_true_positive = np.cumsum(true_positive)
        
        # Precision at each threshold
        precision = cumulative_true_positive / (np.arange(len(cumulative_true_positive)) + 1)
        recall = cumulative_true_positive / num_gt

        
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0.0

# Training and validation loop

def train_and_validate(model, train_loader, test_loader, num_epochs=25):
    train_losses = []
    val_losses = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bbox_targets = torch.tensor([[0, 0, 32, 32]] * images.size(0), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            class_logits, bbox_preds = model(images)

            loss_class = criterion_class(class_logits, labels)
            loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
            loss = loss_class + loss_bbox

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation 
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_pred_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                bbox_targets = torch.tensor([[0, 0, 32, 32]] * images.size(0), dtype=torch.float32).to(device)

                class_logits, bbox_preds = model(images)

                loss_class = criterion_class(class_logits, labels)
                loss_bbox = criterion_bbox(bbox_preds, bbox_targets)
                loss = loss_class + loss_bbox

                val_loss += loss.item()

                all_pred_probs.extend(torch.softmax(class_logits, dim=1).cpu().numpy())
                _, predicted = torch.max(class_logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(test_loader))

 
        # Calculating metrics
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        mean_ap = calculate_map(all_pred_probs, all_labels, num_classes)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}, "
              f"mAP: {mean_ap:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_resnet_object_detector.pth')

    # Saving the model after training and validation
    
    torch.save(model.state_dict(), 'final_resnet_object_detector.pth')
    print("Final model saved as 'final_resnet_object_detector.pth'")
    return train_losses, val_losses

# Plotting loss 

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Main function

def main():
    print("Starting training and validation...")
    train_losses, val_losses = train_and_validate(model, train_loader, test_loader, num_epochs=25)
    plot_loss_curves(train_losses, val_losses)

if __name__ == "__main__":
    main()

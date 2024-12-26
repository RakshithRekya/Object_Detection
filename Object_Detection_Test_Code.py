import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Data transformations for test set

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 test dataset

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model class (same as the one used during training)
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

# Load the saved model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(cifar10_classes)
model = ResNetObjectDetector(num_classes=num_classes)
model.load_state_dict(torch.load('final_resnet_object_detector.pth', map_location=device))
model.to(device)
model.eval()

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
        
       
        precision = cumulative_true_positive / (np.arange(len(cumulative_true_positive)) + 1)
        recall = cumulative_true_positive / num_gt

        # Calculate Average Precision
        
        ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
        average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0.0

# Test the model

def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []
    all_pred_probs = []
    bbox_targets = torch.tensor([[0, 0, 32, 32]]).to(device)  # Dummy bounding box for testing

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            class_logits, bbox_preds = model(images)

            all_pred_probs.extend(torch.softmax(class_logits, dim=1).cpu().numpy())
            _, predicted = torch.max(class_logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    mean_ap = calculate_map(all_pred_probs, all_labels, num_classes)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP: {mean_ap:.4f}")

# Run the evaluation

evaluate_model(model, test_loader)

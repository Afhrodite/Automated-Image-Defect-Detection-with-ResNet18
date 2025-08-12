import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pylab as plt

# For reproducibility
torch.manual_seed(0)

# Config
POSITIVE_DIR = "Positive_tensors"
NEGATIVE_DIR = "Negative_tensors"
BATCH_SIZE = 100
LEARNING_RATE = 0.001
N_EPOCHS = 1


# Custom Dataset
class CustomDataset(Dataset):
    """Custom dataset for loading positive and negative tensor files."""

    def __init__(self, transform=None, train=True):
        positive_files = [
            os.path.join(POSITIVE_DIR, file)
            for file in os.listdir(POSITIVE_DIR)
            if file.endswith(".pt")
        ]
        negative_files = [
            os.path.join(NEGATIVE_DIR, file)
            for file in os.listdir(NEGATIVE_DIR)
            if file.endswith(".pt")
        ]

        num_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * num_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        self.Y = torch.zeros([num_samples], dtype=torch.long)
        self.Y[::2] = 1
        self.Y[1::2] = 0

        if train:
            self.all_files = self.all_files[:30000]
            self.Y = self.Y[:30000]
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]

        self.len = len(self.all_files)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = torch.load(self.all_files[idx])
        label = self.Y[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Model Setup
def create_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=2)
    return model



# Training
def train_model(model, train_loader, validation_loader, criterion, optimizer):
    loss_list = []
    N_test = len(validation_loader.dataset)

    for epoch in range(N_EPOCHS):
        for x_batch, y_batch in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        correct = 0
        for x_val, y_val in validation_loader:
            model.eval()
            y_val_pred = model(x_val)
            _, predicted = torch.max(y_val_pred.data, 1)
            correct += (predicted == y_val).sum().item()

        accuracy = correct / N_test

    return model, loss_list, accuracy


# Misclassified Samples
def get_misclassified_samples(model, validation_loader, limit=4):
    model.eval()
    misclassified_samples = []
    misclassified_preds = []
    misclassified_labels = []
    misclassified_images = []

    with torch.no_grad():
        for idx, (x_val, y_val) in enumerate(validation_loader):
            y_val_pred = model(x_val)
            _, predicted = torch.max(y_val_pred.data, 1)

            misclassified_indices = (predicted != y_val).nonzero(as_tuple=False).view(-1)

            for i in misclassified_indices:
                if len(misclassified_samples) < limit:
                    sample_number = idx * validation_loader.batch_size + i.item()
                    misclassified_samples.append(sample_number)
                    misclassified_preds.append(predicted[i].item())
                    misclassified_labels.append(y_val[i].item())
                    misclassified_images.append(x_val[i])
                else:
                    break
            if len(misclassified_samples) >= limit:
                break

    return misclassified_samples, misclassified_preds, misclassified_labels


# Main
def main():
    print("Loading datasets...")
    train_dataset = CustomDataset(train=True)
    validation_dataset = CustomDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Creating model...")
    model = create_model()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    print("Starting training...")
    start_time = time.time()
    model, loss_list, accuracy = train_model(
        model, train_loader, validation_loader, criterion, optimizer
    )
    print(f"Final Accuracy: {accuracy:.4f}")

    # Plot loss
    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    print("Finding misclassified samples...")
    misclassified_samples, misclassified_preds, misclassified_labels = get_misclassified_samples(
        model, validation_loader
    )

    for i in range(4):
        print(
            f"sample{misclassified_samples[i]} predicted value: tensor([{misclassified_preds[i]}])"
            f"  actual value: tensor([{misclassified_labels[i]}])"
        )


if __name__ == "__main__":
    main()
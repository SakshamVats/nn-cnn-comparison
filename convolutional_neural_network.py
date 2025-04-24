import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import time

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.002
BATCH_SIZE = 256

train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530)) # Can be calculated
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530)) # Can be calculated
])

train_dataset = FashionMNIST(
    root="FashionMNIST/data",
    train=True,
    transform=train_transform,
    download=True
)

test_dataset = FashionMNIST(
    root="FashionMNIST/data",
    train=False,
    transform=test_transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 1x28x28 -> 16x28x28 -> 16x14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16x14x14 -> 32x14x14 -> 32x7x7
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear((128), num_classes)

    def forward(self, x):
        # conv -> batchnorm -> relu -> pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # conv -> batchnorm -> relu -> pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(-1, 32*7*7)
        # fully connected layer 1
        x = F.relu(self.fc1(x))
        # fully connected layer 2 (no relu)
        x = self.fc2(x)
        return x
    
model = CNN(10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step[{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    start_time = time.time()
    best_accuracy = 0.0
    model_save_path = "FashionMNIST/model/cnn_best_model.pth"

    print("Starting training for convolutional NN...")
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        current_accuracy = evaluate(model, device, test_loader, criterion)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            print(f"Saving new best model with accuracy: {best_accuracy:.2f}%")
            torch.save(model.state_dict(), model_save_path)

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds")
    print(f"Best test accuracy achieved by convolutional NN: {best_accuracy:.2f}%")
    
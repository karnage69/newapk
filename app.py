import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class CNN(nn.Module):
    def __init__(self,num_classes=29):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.Dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256,num_classes)
    def forward(self,x):
        x= self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        x= self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)#flattened
        x= self.Dropout(F.relu(self.fc1(x)))
        x= self.fc2(x)
        return(x)
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset= torchvision.datasets.ImageFolder(root=r"D:\newapk\archive\asl_alphabet_train", transform=train_transform)
testset = torchvision.datasets.ImageFolder(root=r"D:\newapk\archive\asl_alphabet_test",transform=test_transform)
trainloader = DataLoader(trainset,batch_size=64, shuffle=True)
testloader = DataLoader(testset,batch_size=64, shuffle=False)
model = CNN().to(device)
criterion= nn.CrossEntropyLoss()
optimize = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler= StepLR(optimize,step_size=5,gamma=0.5)
#lets set up the loop 
# Training parameters
num_epochs = 30
patience = 3
trigger_times = 0
best_val_acc = 0.0
best_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimize.zero_grad()  # You forgot this!
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimize.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(testloader)

    print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

    scheduler.step()

    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    else:
        trigger_times += 1
        print(f"No improvement for {trigger_times} epoch(s)")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break
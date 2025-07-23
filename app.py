import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
from PIL import Image

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define custom dataset class for CSV-based Sign Language MNIST
dataset_path_train = r"D:\newapk\archive\sign_mnist_train\sign_mnist_train.csv"
dataset_path_test = r"D:\newapk\archive\sign_mnist_test\sign_mnist_test.csv"

class SignLanguageMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        # Remove rows with label 9 (J) and 25 (Z)
        data = data[~data['label'].isin([9, 25])].reset_index(drop=True)
        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)
        img = np.stack([img] * 3, axis=-1)  # Grayscale to RGB
        label = self.labels[idx]
        # Remap labels to be 0 to 23
        if label > 9:
            label -= 1  # Skip 'J'
        if label > 25:
            label -= 1  # Skip 'Z' (just in case)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
trainset = SignLanguageMNISTDataset(dataset_path_train, transform=train_transform)
testset = SignLanguageMNISTDataset(dataset_path_test, transform=test_transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define class labels (0-25, excluding J and Z due to motion)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
idx_to_class = {i: label for i, label in enumerate(class_labels)}

# CNN model class
class CNN(nn.Module):
    def __init__(self, num_classes=24):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize model, loss, optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_loss = float('inf')
patience = 3
trigger_times = 0

for epoch in range(20):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    val_loss, val_correct, val_total = 0.0, 0, 0
    model.eval()
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

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# Load model for webcam inference
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Webcam transform
webcam_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Webcam prediction loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

buffered_letters = ""
prev_letter = ""

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    size = 200
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_tensor = webcam_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        predicted_label = idx_to_class.get(predicted.item(), '?')

    if confidence.item() > 0.75 and predicted_label != prev_letter:
        buffered_letters += predicted_label
        prev_letter = predicted_label

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'Letter: {predicted_label}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f'Word: {buffered_letters}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('ASL Detection - Press q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

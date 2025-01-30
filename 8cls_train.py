import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import shufflenet_v2_x1_0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')

def get_dataloader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip() if train else transforms.Compose([]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader

def train_model(model, train_loader, val_loader, num_epochs=70):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model_weights_V5.pth')
            print(f'Saved Best Model with Accuracy: {accuracy}%')

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%')

model = shufflenet_v2_x1_0(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 8)
model.to(device)

train_loader = get_dataloader(r'D:\myfiles\school\Autodrive\traffic_light_classifier\AutoDrive\8classes_data\train', train=True)
val_loader = get_dataloader(r'D:\myfiles\school\Autodrive\traffic_light_classifier\AutoDrive\8classes_data\val', train=False)
train_model(model, train_loader, val_loader)
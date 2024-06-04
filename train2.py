from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.nn import BCEWithLogitsLoss
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Training on {device}')

# 准备多标签的binarizer
mlb = MultiLabelBinarizer(classes=range(6))  # 假设有6个类别
mlb.fit([range(6)])  # 预设所有可能的类别标签

def get_dataloader(data_dir, batch_size=32, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataset.samples = [(s, mlb.transform([[int(l)]])[0]) for s, l in dataset.samples]  # 将标签转换为多标签二进制形式

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader


def train_model(model, train_loader, val_loader, num_epochs=50):
    model = model.to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

    torch.save(model.state_dict(), 'best_model_weights.pth')  # Save the model

# Modify the classifier for 6 classes
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)  # Assuming there are 6 classes
model.to(device)

train_loader = get_dataloader(r'D:\myfiles\school\Autodrive\traffic_light_classifier\train', train=True)
val_loader = get_dataloader(r'D:\myfiles\school\Autodrive\traffic_light_classifier\val', train=False)
train_model(model, train_loader, val_loader)

def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('best_model_weights.pth'))  # Load the best model weights
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            outputs = torch.sigmoid(outputs) > 0.5  # Applying threshold to get binary outputs
            correct += (outputs == labels).all(dim=1).sum().item()
            total += labels.size(0)

    print(f'Accuracy on test set: {100 * correct / total}%')

test_loader = get_dataloader(r'D:\myfiles\school\Autodrive\traffic_light_classifier\test', train=False)
evaluate_model(model, test_loader)

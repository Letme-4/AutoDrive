import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)

model.load_state_dict(torch.load('path'))
model.to(device)
model.eval()

# preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).int()
    return probabilities.cpu().numpy()[0], predictions.cpu().numpy()[0]

probabilities, predicted_classes = predict_image(model, r"traffic_light_path")
print(f'Predicted probabilities: {probabilities}')
print(f'Predicted classes: {predicted_classes}')

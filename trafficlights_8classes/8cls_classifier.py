import torch
from torchvision.models import shufflenet_v2_x1_0
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time

device = torch.device("cpu")

model = shufflenet_v2_x1_0(pretrained=False)
num_classes = 8 # adjust it depends on situation
'''
green
green_left
green_yellowleft
not_green
not_green_left
notgreen_greenleft
notgreen_yellowleft
null
'''
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(r'D:\myfiles\school\Autodrive\traffic_light_classifier\AutoDrive\tools\best_model_weights_V5.pth', map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.9).int()
    elapsed_time = (time.time() - start_time) * 1000

    return probabilities.cpu().numpy()[0], predictions.cpu().numpy()[0], elapsed_time

probabilities, predicted_classes, time_ms = predict_image(model, r"D:\myfiles\school\Autodrive\traffic_light_classifier\AutoDrive\newdata\test\06.png")
print(f'Predicted probabilities: {probabilities}')
print(f'Predicted classes: {predicted_classes}')
print(f'Processing time: {time_ms:.2f} ms')

from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch

app = Flask(__name__)

def load_model():
    model = models.resnet152()
    model.fc = nn.Linear(model.fc.in_features, 196)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

model = load_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    input_tensor = transform_image(image).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output).item()
    return jsonify({'prediction': prediction})

@app.route('/')
def index():
    return open('index.html').read()

if __name__ == '__main__':
    app.run(debug=True)

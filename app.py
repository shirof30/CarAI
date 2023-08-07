from flask import Flask, send_file, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch

app = Flask(__name__)

class CarModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._build_transform()

    def _load_model(self):
        resnet_model = models.resnet152()
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 196)
        state_dict = torch.load('model.pth')
        resnet_model.load_state_dict(state_dict)
        resnet_model.eval()
        return resnet_model.to(self.device)

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = torch.argmax(output).item()
        return prediction

car_model = CarModel()

@app.route('/predict', methods=['POST'])
def predict_image():
    file_data = request.files['file']
    image_data = Image.open(file_data.stream).convert('RGB')
    result = car_model.predict(image_data)
    return jsonify({'prediction': result})

@app.route('/')
def main_page():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)

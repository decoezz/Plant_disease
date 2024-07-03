from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from io import BytesIO

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = r'C:\Users\decoe\Desktop\Comptition\Model'
model = ViTForImageClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Function to predict plant disease from image bytes
def predict_plant_disease(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    
    # Get the predicted class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
    
    return predicted_class.item(), probabilities[0][predicted_class].item()

# Initialize Flask app
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    predicted_class, confidence = predict_plant_disease(image_bytes)
    
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

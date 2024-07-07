from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from io import BytesIO

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = r'/root/Plant_disease/Model'
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
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)

    # Get the predicted class
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    print(f'Logits: {logits}')
    print(f'Probabilities: {probabilities}')
    print(f'Predicted Class Index: {predicted_class}')

    return predicted_class, probabilities[0][predicted_class].item()

# Disease information
def disease_information(predicted_class):
    diseases = {
        0: {
            'name': 'Alternaria Leaf Spot',
            'description': 'Alternaria leaf spot is characterized by small, dark, circular spots with concentric rings (target-like spots) on leaves, which can cause leaf yellowing and drop. In severe cases, the spots can coalesce, leading to large dead areas on the leaf.',
            'conditions': 'Warm and humid conditions favor the growth and spread of Alternaria fungi. Poor air circulation can exacerbate the problem, as it keeps the foliage wet for extended periods. High moisture levels, including heavy dew, rainfall, or overhead irrigation, can promote fungal growth.',
            'mechanical_treatment': 'Remove and destroy infected plant debris to reduce sources of inoculum. Ensure good air circulation around plants by proper spacing and pruning. Water plants at the base to avoid wetting foliage and reduce leaf wetness duration.',
            'chemical_treatment': 'Chlorothalonil:Usage: Apply every 7-10 days during conditions favorable for disease development.Application: Follow label instructions for specific crops and timing.Effectiveness: Controls a broad spectrum of fungal pathogens, including Alternaria.Copper-Based Fungicides (e.g., Copper Hydroxide, Copper Oxychloride):Usage: Apply at the first sign of disease and repeat applications every 7-10 days as necessary.Application: Use according to the manufacturer\'s instructions.Effectiveness: Provides protection against many fungal and bacterial diseases.Mancozeb:Usage: Apply as a preventative treatment or at the first sign of symptoms.Application: Follow label directions for dosage and application frequency.Effectiveness: Effective against a variety of fungal pathogens, including Alternaria.Azoxystrobin:Usage: Apply as a foliar spray, typically every 7-14 days depending on disease pressure.Application: Use according to the label instructions for specific crops.Effectiveness: A systemic fungicide that provides both protective and curative action against Alternaria.Difenoconazole:Usage: Apply as a foliar spray at the first sign of disease.Application: Follow the label for specific crops and timing.Effectiveness: Effective in controlling Alternaria and other fungal pathogens.',
            'source': 'https://en.wikipedia.org/wiki/Alternaria_leaf_spot'
        },
        # Add other diseases here with their respective indices
    }
    
    return diseases.get(predicted_class, {'error': 'Disease information not available'})

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
    disease_info = disease_information(predicted_class)
    
    response = {
        'disease': disease_info,
        'confidence': confidence
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

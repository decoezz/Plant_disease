import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import requests
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

# Function to predict plant disease from URL
def predict_plant_disease_from_url(image_url):
    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
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

# Get image URL from the user
image_url = input("Enter the URL to the plant image: ")

# Test the function
predicted_class, confidence = predict_plant_disease_from_url(image_url)
print(f'Predicted class: {predicted_class}, Confidence: {confidence:.4f}')

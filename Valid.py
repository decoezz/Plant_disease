import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from torch.utils.tensorboard import SummaryWriter

# Define a wrapper to ensure the model returns a tuple instead of a dict
class WrappedViTForImageClassification(torch.nn.Module):
    def __init__    (self, model):
        super(WrappedViTForImageClassification, self).__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output.logits  # Return logits directly to avoid dict output

# Load the trained model
model_path = r'C:\Users\decoe\Desktop\Comptition\Model'
model = ViTForImageClassification.from_pretrained(model_path)
model = WrappedViTForImageClassification(model)
model.eval()  # Set the model to evaluation mode

# Define transformations and load validation dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_path = r'C:\Users\decoe\Desktop\Comptition\Validation'
val_dataset = datasets.ImageFolder(val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Set up TensorBoard writer
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Add the model graph to TensorBoard
dummy_input = torch.zeros(1, 3, 224, 224).to(device)  # Create a dummy input tensor
writer.add_graph(model, dummy_input)

# Evaluation loop
print("Evaluating the model...")
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute average loss and accuracy
val_loss /= len(val_loader)
val_accuracy = 100.0 * correct / total

print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

# Log metrics to TensorBoard
writer.add_scalar('Loss/validation', val_loss)
writer.add_scalar('Accuracy/validation', val_accuracy)

# Close the TensorBoard writer
writer.close()

print("Evaluation and logging completed.")

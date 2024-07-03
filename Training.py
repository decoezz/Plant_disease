import os
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Disable the symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Verify GPU availability
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(f"CUDA available: {cuda_available}")
print(f"Number of GPUs: {num_gpus}")
if cuda_available:
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Define number of classes and epochs
num_classes = 40
num_epochs = 20

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths
train_path = r'C:\Users\decoe\Desktop\Comptition\Train'
val_path = r'C:\Users\decoe\Desktop\Comptition\Validation'

# Check if directories exist
print(f"Training directory exists: {os.path.exists(train_path)}")
print(f"Validation directory exists: {os.path.exists(val_path)}")

if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError(f"One or both of the specified directories do not exist: \nTrain: {train_path}\nValidation: {val_path}")

# Load your dataset
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

# Ensure that labels are within the expected range
train_labels = [label for _, label in train_dataset.samples]
val_labels = [label for _, label in val_dataset.samples]

assert all(0 <= label < num_classes for label in train_labels), "Train labels are out of range."
assert all(0 <= label < num_classes for label in val_labels), "Validation labels are out of range."

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if cuda_available else "cpu")
print(f"Using device: {device}")

# Load pre-trained model with ignore_mismatched_sizes
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True)
model.to(device)

# Confirm which GPU is being used
if device.type == 'cuda':
    current_device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(current_device)}")
else:
    print("Using CPU")

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Set up TensorBoard writer
log_dir = "./logs"  # Specify the directory where you want to store TensorBoard logs
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Training loop
print("Training started...")
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the specified device
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.logits, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Compute average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100.0 * total_correct / total_samples
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
   
    # Add scalars to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, global_step=epoch)
    writer.add_scalar('Accuracy/train', epoch_accuracy, global_step=epoch)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the specified device
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
    
    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / total
    # Add scalars to TensorBoard
    writer.add_scalar('Loss/validation', val_loss, global_step=epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, global_step=epoch)
    
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
     
    

# Close the TensorBoard writer
writer.close()

# Save the fine-tuned model
model.save_pretrained(r'C:\Users\decoe\Desktop\Comptition\Save')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training and saving completed. Total time: {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.0f} seconds")

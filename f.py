import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Load the pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()

# Define a function to preprocess images for the model
def preprocess_image(image_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_data).convert("RGB")
    return transform(image).unsqueeze(0)

# Function to predict class labels using the pre-trained model
def predict_image(image_data):
    image_tensor = preprocess_image(image_data)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Example: Assuming you have an image stored in 'image_medium' field of a product
product_id = 1  # Replace with your product ID
product = self.pool['product.product'].browse(cr, uid, product_id)
image_data = product.image_medium
predicted_class = predict_image(image_data)
print(f"Predicted Class for Product {product.name}: {predicted_class}")

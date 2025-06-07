import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import streamlit as st

# --- Model Definition ---
class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # remove last fc
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# --- Load Model ---
st.cache_data
def load_model():
    model = ResNetWithDropout(num_classes=2)  # Update with your class count
    torch.save(model.state_dict(), "ecg_model.pth")
    model.eval()
    return model

model = load_model()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- Web App UI ---
st.title("ECG Image Classification")
st.write("Upload an ECG image to predict the abnormality class")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["Myocrdial Infarction", "Normal"]  # Your classes
    class_descriptions = {
        "Myocrdial Infarction": "A condition where blood flow to a part of the heart is blocked, causing heart muscle damage.",
        "Normal": "No abnormality detected in the ECG image; the heart appears healthy."
    }

    st.write(f"### Predicted Class: {class_names[predicted.item()]}")
    st.write(f"*Description:* {class_descriptions[class_names[predicted.item()]]}")
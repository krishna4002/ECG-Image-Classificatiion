# 🫀 ECG Image Classification using ResNet

This project is a simple yet powerful web application that allows users to **upload ECG images and get instant predictions** on whether the ECG is **normal** or indicates a **myocardial infarction (heart attack)**. The backend is built with **PyTorch**, using a customized **ResNet-18** deep learning model, while the frontend uses **Streamlit** for a smooth, interactive experience.

---

## About the Project

This project was developed to showcase the integration of:

- **Deep Learning for healthcare**
- **Image preprocessing and classification**
- **Web application using Streamlit**

The application helps demonstrate how machine learning can assist in medical diagnostics. It's designed to classify ECG images into two categories:

1. **Normal** — No visible abnormalities in the ECG.
2. **Myocardial Infarction (MI)** — A serious heart condition caused by blocked blood flow.

The model is trained (or placeholder initialized) using PyTorch and uses a ResNet-18 CNN architecture with added dropout to prevent overfitting.

---

## Key Features

- **Pre-trained Deep Learning Model**: A ResNet-18 neural network adapted for ECG image classification.
- **Dropout Regularization**: Enhances generalization and reduces overfitting.
- **User Image Upload**: Upload ECG images in JPG, JPEG, or PNG format.
- **Instant Prediction Output**: The app instantly shows the predicted class along with a short medical explanation.
- **Web Interface**: Built with Streamlit, no need for a complex UI setup.

---

## How Users Can Interact with It

1. Open the web app (via `streamlit run app.py`).
2. Click "Choose an image" and upload your ECG image file.
3. The model processes the image, runs a prediction, and displays:
   - The **predicted class** (Normal or Myocardial Infarction)
   - A **short description** explaining the result

This makes it easy for users (even non-developers) to interact with AI-powered medical predictions.

---

## Example Output

Suppose a user uploads an ECG image:

- **Predicted Class**: Myocardial Infarction  
- **Explanation**: A condition where blood flow to a part of the heart is blocked, causing heart muscle damage.

---

## Project Structure

```
ecg-classification/
│
├── app.py                  # Streamlit app that runs the entire pipeline
├── ecg_model.pth           # Saved model weights (created during execution)
├── ECG_work_notebook.ipynb # Jupyter notebook used to develop and test the model
├── requirements.txt        # List of required Python libraries
```

---

## Installation Guide

> This project is tested with Python 3.8+.

### 1. Clone the repository (optional)

```bash
git clone https://github.com/your-username/ECG-Image-Classification.git
cd ECG-Image-Classification
```

### 2. Install all required packages

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The Streamlit server will open in your browser where you can upload ECG images.

---

## requirements.txt

```txt
# Core dependencies
streamlit           # For the frontend web app
torch               # For deep learning model
torchvision         # For using pretrained ResNet model
Pillow              # For image processing

# Optional for development
notebook            # For working in Jupyter
matplotlib          # For data visualization (in notebook)
```

---

## How It Works - Behind the Scenes

1. **Model Definition**: A PyTorch model based on ResNet-18 is defined in `app.py`.
2. **Dropout Layer**: Added for regularization to prevent overfitting.
3. **Image Preprocessing**: Uploaded ECG images are resized and converted into tensors using `torchvision.transforms`.
4. **Model Prediction**: The model runs inference and picks the class with the highest confidence.
5. **Result Display**: Prediction label and explanation are shown in the UI using Streamlit.

---

## Future Improvements

- Train and load actual model weights for better accuracy.
- Expand classification to detect more cardiac conditions.
- Add confidence scores and visual overlays (e.g., Grad-CAM).
- Deploy the app on platforms like Streamlit Cloud or Hugging Face Spaces.

---

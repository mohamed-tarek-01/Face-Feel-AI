# Face Feel AI - Facial Emotion Recognition using EfficientNet

## 🧠 Domain: Computer Vision & Deep Learning

This project focuses on developing a facial emotion recognition (FER) system using the FER2013 dataset. The model classifies facial expressions into seven basic emotions, helping machines understand human affective states from facial images.


## 🎯 Project Objective

To build a deep learning model that can accurately recognize human emotions from facial expressions using grayscale images. The model aims to support applications in mental health analysis, user experience enhancement, and human-computer interaction.


## 🧰 Model Overview

- **Architecture**: EfficientNetV2S (Transfer Learning)
- **Why EfficientNetV2S?**: EfficientNetV2S achieves a better trade-off between accuracy and efficiency compared to traditional CNNs. It leverages pretrained ImageNet weights, enabling faster convergence and improved performance on relatively small datasets like FER2013.

### ✅ Key Features:

- Pretrained EfficientNetV2S as base model (ImageNet weights)
- Global Average Pooling for dimensionality reduction
- Dropout layers for regularization
- Dense layers for final classification into 7 emotions
- Trained with augmented data to enhance generalization

## 📦 Dataset

- **Name**: [FER2013 - Facial Expression Recognition](https://www.kaggle.com/datasets/msambare/fer2013)
- **Emotions**:
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral
- **Image Format**: 48x48 grayscale images
- **Preprocessing**:
  - Normalization and resizing
  - One-hot encoding of emotion labels
  - Train/validation split

### 🧪 Data Augmentation:
- Rotation (simulate head tilts)  
- Horizontal flipping (mirror faces)  
- Brightness adjustments (light variations)  
- Zoom & cropping (focus on face area)

## 📈 Model Training & Evaluation

- **Framework**: TensorFlow / Keras
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Final Accuracy**: ~68% on the FER2013 test set

## 🖥️ Flask Web Application

A lightweight web interface was built using Flask to deploy the trained model and allow users to test it interactively.

### 🎯 Features of the Flask App

- **📸 Upload Image**: Users can upload a facial image to detect emotion.
- **⚙️ Real-time Prediction**: Model processes and classifies the image instantly.
- **🎨 Simple UI**: HTML, CSS, and JS provide a user-friendly experience.
- **📂 Local Hosting**: Easy to run and test on any machine.

## 📁 Repository Structure

```
Face Feel AI/
├── app.py # Flask backend
├── README.md # Project documentation
├── models/
│   └── model.tflite # Trained TFLite model
├── notebooks/
│   └── Facial Emotion Recognition.ipynb # Jupyter Notebook for training
├── static/
│   ├── favicon.ico # Favicon displayed in browser tab
│   ├── Main-1.jpeg # Project logo
│   ├── styles.css # CSS for UI
│   └── script.js # JS for interactivity
├── templates/
│   ├── index.html # Home page
│   ├── about_me.html # About the developer
│   └── our_work.html # Project explanation

```

## 🚀 Run the App

To run the Flask app locally:

```bash
python app.py
```
Then open your browser at http://localhost:5000

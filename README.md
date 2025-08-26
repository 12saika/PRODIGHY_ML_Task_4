#Prodighy Task 04


This project implements a hand gesture recognition system using ResNet50 pretrained on ImageNet and fine-tuned on the Leap Motion Gesture Recognition Dataset
. The model classifies different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

🚀 Features

✅ Transfer learning with ResNet50 (PyTorch)

✅ Custom classification layer for gesture recognition

✅ Data preprocessing & normalization (resize, tensor conversion, ImageNet mean/std)

✅ Training & validation with performance tracking

✅ GPU acceleration using CUDA

✅ Applications in AR/VR, robotics, and sign language recognition

📂 Dataset

Source: Leap Motion Gesture Dataset

Description: Images of multiple hand gestures captured using a Leap Motion sensor

Classes: Multiple gesture categories (swipes, rotations, etc.)

🛠️ Tech Stack

Language: Python

Framework: PyTorch, TorchVision

Libraries: NumPy, Matplotlib, PIL

📊 Results

High validation accuracy across gesture classes

Robust recognition suitable for real-world HCI applications

⚡ Usage
1️⃣ Clone the repository
git clone [https://github.com/your-username/gesture-recognition.git](https://github.com/12saika/PRODIGHY_ML_Task_4.git)
cd gesture-recognition

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python train.py

4️⃣ Run inference on a new image
from PIL import Image
import torch
from torchvision import transforms

# Load model
model = torch.load("resnet50_gesture.pth")
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
img = Image.open("sample.jpg")
img_t = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img_t)
    _, pred = torch.max(output, 1)

print("Predicted class:", pred.item())

🎯 Applications

AR/VR gesture control

Human-Computer Interaction (HCI)

Robotics interfaces

Sign language recognition

📌 Author

👨‍💻 Your Name Saika Parvin
🔗 LinkedIn - https://www.linkedin.com/in/saika-parvin-865061362/
 | GitHub  -  https://github.com/12saika

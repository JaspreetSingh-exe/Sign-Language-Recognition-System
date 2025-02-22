# Sign Language Recognition using MediaPipe & MLP

## 📌 Project Overview

This project implements **real-time sign language recognition** using **MediaPipe Hands** for hand landmark detection and an **MLP (Multi-Layer Perceptron) model** for character classification. Additionally, we experimented with **MobileNetV2** for sign recognition but found that **MediaPipe-based MLP performs better in real-time scenarios**. The system captures hand gestures from a webcam, extracts landmark features, and predicts sign language letters, dynamically forming words and sentences.

## 📂 Project Structure

```
📂 Sign Language Recognition System/
│── 📂 SIGN_TO_SENTENCE_PROJECT/
│    │── 📂 Asl_Sign_Data/                       # Raw ASL dataset
│    │── 📄 asl_mediapipe_keypoints_dataset.csv  # Preprocessed dataset for MLP model
│    │── 📄 asl_mediapipe_mlp_model.h5           # Trained MLP model
│    │── 📄 sign_language_model_MobileNetV2.h5   # Trained MobileNetV2 model
│    │── 📄 Combined_Architecture.ipynb          # Hybrid model experiments
│    │── 📄 LLM.ipynb                            # Language Model Integration
│    │── 📄 Mediapipe_Training.ipynb             # Training script for MLP model
│    │── 📄 MobileNetV2_Training.ipynb           # Training script for MobileNetV2
│    │── 📄 concluion.txt                        # Summary of results
│    │── 📄 requirements.txt                     # Required dependencies
```

## 🏗️ Dataset: ASL Kaggle Dataset

- The dataset used for training was obtained from **Kaggle ASL Sign Language Dataset**.
- It contains **hand gesture images labeled with ASL characters**.
- For **MobileNetV2**, we used **raw images**.
- For **MLP (MediaPipe)**, we extracted **landmark keypoints** from each image and stored them in CSV format.

## 📌 Model 1: MobileNetV2 Approach

- We trained **MobileNetV2** on raw images for sign classification.
- However, it struggled with **real-time sign recognition**, leading us to explore **MediaPipe-based MLP**.

### **Training MobileNetV2**

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Add custom classification layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## 📌 Model 2: MediaPipe + MLP Approach

- Extracted **hand landmark coordinates** using **MediaPipe Hands**.
- Trained a **MLP model** on the extracted landmark features.
- This method proved to be **faster and more reliable for real-time recognition**.

### **Training MLP on MediaPipe Landmarks**

```python
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained MLP model
mlp_model = tf.keras.models.load_model("asl_mediapipe_mlp_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Predict a sign using MediaPipe landmarks
def predict_sign(landmarks):
    input_data = np.array(landmarks).flatten().reshape(1, -1)
    prediction = mlp_model.predict(input_data)
    return np.argmax(prediction)
```

## 🚀 Running the Sign Language Recognition System

### **1️⃣ Install Dependencies**

```sh
pip install -r requirements.txt
```

### **2️⃣ Running Model Evaluations**

To test the output of individual models, run the **last cell** in:

- `Mediapipe_Training.ipynb` for MLP model evaluation.
- `MobileNetV2_Training.ipynb` for MobileNetV2 evaluation.

### **3️⃣ Running the Combined Architecture**

To see the working of **both MobileNetV2 and MediaPipe integrated**, run:

```sh
jupyter notebook Combined_Architecture.ipynb
```

### **4️⃣ Controls & Commands**

- **Normal Signs** → Letters are appended to the sentence.
- **SPACE Sign** → Adds a space.
- **DELETE Sign** → Removes the last character.
- **NOTHING** → No input detected.

## ⚠️ Limitations & Next Steps

- **MobileNet did not perform well on real-time images**, so we moved to **MediaPipe-based MLP**.
- **Next Phase** → Building a FastAPI backend (`SignConnect-Backend`) for better integration and mobile app support.

## 🔜 Future Work: Text-to-Sign Language Generation

As the next phase of development, we aim to implement **Text-to-Sign Language Actions**, allowing users to input text that gets translated into sign language animations. Possible technologies we will explore:
- **AI-generated 3D avatars** to perform sign language gestures.
- **Computer Vision & Reinforcement Learning** to map text to sign movements.
- **Deep Learning models** to generate smooth sign transitions.

### 💡 Open for Contributions
We welcome contributions from the community for this phase! If you're interested in helping develop **Text-to-Sign Language Generation**, feel free to open an issue or submit a pull request on our GitHub repository.


## 🤝 Acknowledgments

- Uses **MediaPipe Hands** for landmark detection.
- Model trained using **TensorFlow & Scikit-Learn**.
- Inspired by existing research on **gesture recognition & sign language AI**.


## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.  

---



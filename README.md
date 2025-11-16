# AI-Deepfake-Detection

## 🎯 Objective
This project aims to **detect DeepFake images** using deep learning.  
By identifying fake media early, it helps reduce the spread of misinformation.

---

## 🧰 Tech Stack
- Python 3.11  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas, Matplotlib  
- Streamlit (for the web app)

---

## 📂 Project Structure
DeepFake-Detection-and-Prevention-AI/
│
├── data/ # Dataset folders (train, val, test)
├── models/ # Saved trained models (.h5 files)
├── sample/ # Example images for testing
├── app.py # Streamlit app for UI
├── data_generator.py # Data augmentation and preprocessing
├── split_dataset.py # Splits dataset into train/val/test
├── train_model.py # Model building and training
├── predict.py # Predicts single image
├── debug_predict.py # Debugging and CSV predictions
└── README.md # Project documentation


---

## 🧬 Model Details
- Base Model: **MobileNetV2** (pre-trained on ImageNet)
- Added Layers: Dense + Dropout for binary classification  
- Loss Function: Binary Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy  

Data Augmentation includes:
- Rotation, Zoom, Brightness, Shear, Horizontal Flip  
- MobileNetV2 preprocessing for normalization

---

## 🧪 Dataset
Images are divided into:
- **train/** — used to train model  
- **val/** — used to tune hyperparameters  
- **test/** — used for final evaluation  

You can use any DeepFake image dataset such as:
- [Kaggle DeepFake Detection Dataset](https://www.kaggle.com/c/deepfake-detection-challenge)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

---

## ⚙️ How to Run the Project

### Step 1: Activate virtual environment
```bash
.\venv\Scripts\activate

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Train the model
python train_model.py

Step 4: Run Streamlit app
streamlit run app.py

| Metric              | Value   |
| ------------------- | ------- |
| Training Accuracy   | ~85%    |
| Validation Accuracy | ~72%    |
| Test Accuracy       | ~68–70% |

🚀 Future Improvements

Add detection for DeepFake videos, not just images.

Improve accuracy using EfficientNet or XceptionNet.

Deploy to cloud platforms.

Add watermark or authenticity verification system.

## 📊 Training Results

### Accuracy
![Accuracy](sample/plots/accuracy.png)

### Loss
![Loss](sample/plots/loss.png)

### Confusion Matrix
![Confusion Matrix](sample/plots/confusion_matrix.png)

### ROC Curve
![ROC Curve](sample/plots/roc_curve.png)

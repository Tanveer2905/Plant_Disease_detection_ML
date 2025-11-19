🌱 Plant Disease Detection & Segmentation (Flask App)

A complete end-to-end Deep Learning + Machine Learning pipeline for detecting plant leaf diseases and generating segmentation masks, histograms, and multi-model predictions.
This project uses 7 classification models, background removal, U-Net segmentation, and a Flask web interface + API.

🔥 Key Features
✅ 1. Multi-Model Classification (Majority Voting)

The system predicts plant disease using 7 models:

CNN (custom)

ResNet

MobileNet

VGG16

XGBoost

Random Forest

SVM

Final prediction = majority vote across all models.

✅ 2. Leaf Segmentation

Background removed using Rembg

Segmentation mask generated using
leaf_disease_segmentation_model.h5

Output returned as base64 PNG

✅ 3. Pixel Histogram Visualization

Generates RGB pixel distribution graphs with 95th percentile scaling to avoid spikes.

✅ 4. Full Flask Web App

Includes:

/ → Upload page

/result → Prediction + mask + histogram

/predict_api → JSON response for mobile apps

CORS enabled for Android integration

Runs on Waitress for production

✅ 5. Clean UI (upload.html + result.html)

(Renders uploaded image, predictions, segmentation mask, histogram)

📥 Download Model Files

GitHub does not allow 100MB+ files, so all models are provided in a ZIP file:

🔗 Download Models (Google Drive):

https://drive.google.com/file/d/14wbawlZa7VuUVRm2D_iq4nmvrW1tUAng/view?usp=sharing

After downloading:

unzip model.zip
place all extracted files inside: /models/


Expected model files:

models/
│── cnn_model.keras
│── resnet_model.keras
│── mobilenet_model.keras
│── final_vgg16_model.h5
│── xgb_model.pkl
│── random_forest.pkl
│── svm_model.pkl

📂 Project Structure
Plant-Disease-Detection/
│── predict.py
│── models/
│── dataset/
│── leaf_disease_segmentation_model.h5
│── static/
│    └── uploads/
│── templates/
│    ├── upload.html
│    └── result.html
│── README.md

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

3. Install dependencies
pip install -r requirements.txt

▶️ Running the Application

The application uses Waitress (production server):

python predict.py


You will see:

Server running at: http://YOUR_LOCAL_IP:5000


Open browser:

http://127.0.0.1:5000

🧪 API Endpoint (For Android / Mobile Apps)
POST /predict_api

Form-Data:

file: image.jpg

Response JSON:
{
  "predictions": {
    "CNN Prediction": "...",
    "ResNet Prediction": "...",
    "MobileNet Prediction": "...",
    "VGG16 Prediction": "...",
    "XGBoost Prediction": "...",
    "Random Forest Prediction": "...",
    "SVM Prediction": "...",
    "Final Prediction": "..."
  },
  "segmentation_mask": "<base64-png>",
  "pixel_histogram": "<base64-png>"
}

📊 How the System Works Internally
1. Preprocessing

Image resized to 128×128

Normalized to 0–1

2. Classification Pipeline

Each model predicts a label index

FeatureExtractor (CNN backbone) used for ML models

Majority vote selects final prediction

3. Segmentation Pipeline

Background removal using rembg

Image resized to 256×256

Segmentation model generates mask

Mask converted to base64 PNG

4. Histogram

Uses OpenCV to plot RGB histograms with statistical clipping.

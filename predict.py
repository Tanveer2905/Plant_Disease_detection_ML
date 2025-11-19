import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from collections import Counter
from flask import Flask, request, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained models
cnn_model = load_model('models/cnn_model.keras')
resnet_model = load_model('models/resnet_model.keras')
mobilenet_model = load_model('models/mobilenet_model.keras')
vgg16_model = load_model('models/final_vgg16_model.h5')

xgb_model = joblib.load('models/xgb_model.pkl')
rf_model = joblib.load('models/random_forest.pkl')
svm_model = joblib.load('models/svm_model.pkl')

# Load class labels dynamically
train_set = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    'dataset/train', target_size=(128, 128), batch_size=32, class_mode='sparse', shuffle=False)
class_labels = {i: label for label, i in train_set.class_indices.items()}

# Initialize models with dummy input
dummy_input = np.zeros((1, 128, 128, 3))
cnn_model.predict(dummy_input)
resnet_model.predict(dummy_input)
mobilenet_model.predict(dummy_input)
vgg16_model.predict(dummy_input)

# Feature extraction for ML models
input_shape = (128, 128, 3)
dummy_tensor_input = tf.keras.Input(shape=input_shape)
features = cnn_model(dummy_tensor_input)
feature_extractor = Model(inputs=dummy_tensor_input, outputs=features)

def preprocess_image(image_path):
    """Preprocesses an image for classification models."""
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def get_predictions(image_path):
    """Predicts plant disease using all models including VGG16."""
    img_array = preprocess_image(image_path)
    
    # Deep Learning Model Predictions
    cnn_pred = np.argmax(cnn_model.predict(img_array), axis=1)[0]
    resnet_pred = np.argmax(resnet_model.predict(img_array), axis=1)[0]
    mobilenet_pred = np.argmax(mobilenet_model.predict(img_array), axis=1)[0]
    vgg16_pred = np.argmax(vgg16_model.predict(img_array), axis=1)[0]
    
    # Extract features for ML models
    features = feature_extractor.predict(img_array).flatten().reshape(1, -1)
    xgb_pred = xgb_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]
    
    # Majority Voting for Final Prediction
    predictions = [cnn_pred, resnet_pred, mobilenet_pred, vgg16_pred, xgb_pred, rf_pred, svm_pred]
    final_prediction = Counter(predictions).most_common(1)[0][0]

    # Print predictions to terminal
    print("\n--- Model Predictions ---")
    print(f"CNN Prediction: {class_labels.get(cnn_pred, 'Unknown')}")
    print(f"ResNet Prediction: {class_labels.get(resnet_pred, 'Unknown')}")
    print(f"MobileNet Prediction: {class_labels.get(mobilenet_pred, 'Unknown')}")
    print(f"VGG16 Prediction: {class_labels.get(vgg16_pred, 'Unknown')}")
    print(f"XGBoost Prediction: {class_labels.get(xgb_pred, 'Unknown')}")
    print(f"Random Forest Prediction: {class_labels.get(rf_pred, 'Unknown')}")
    print(f"SVM Prediction: {class_labels.get(svm_pred, 'Unknown')}")
    print(f"FINAL Prediction: {class_labels.get(final_prediction, 'Unknown')}\n")

    return {
        "CNN Prediction": class_labels.get(cnn_pred, "Unknown"),
        "ResNet Prediction": class_labels.get(resnet_pred, "Unknown"),
        "MobileNet Prediction": class_labels.get(mobilenet_pred, "Unknown"),
        "VGG16 Prediction": class_labels.get(vgg16_pred, "Unknown"),
        "XGBoost Prediction": class_labels.get(xgb_pred, "Unknown"),
        "Random Forest Prediction": class_labels.get(rf_pred, "Unknown"),
        "SVM Prediction": class_labels.get(svm_pred, "Unknown"),
        "Final Prediction": class_labels.get(final_prediction, "Unknown")
    }

from rembg import remove
from PIL import Image
import io

def segment_image(image_path):
    """Removes background, generates segmentation mask and histogram, and returns base64-encoded results."""

    # Step 1: Remove background
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
    output_data = remove(input_data)

    # Convert to a PIL image and then to OpenCV format
    pil_image = Image.open(io.BytesIO(output_data)).convert("RGB")
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Step 2: Preprocess for segmentation model
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)

    # Load and predict with segmentation model
    segmentation_model = load_model("leaf_disease_segmentation_model.h5")
    pred_mask = segmentation_model.predict(img_batch)[0].squeeze()

    # --- Convert mask to base64 ---
    fig_mask, ax_mask = plt.subplots()
    ax_mask.imshow(pred_mask, cmap='jet')
    ax_mask.axis('off')
    buf_mask = io.BytesIO()
    plt.savefig(buf_mask, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig_mask)
    buf_mask.seek(0)
    mask_base64 = base64.b64encode(buf_mask.getvalue()).decode('utf-8')

    # --- Pixel value distribution ---
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    colors = ('r', 'g', 'b')
    channel_labels = ('Red Channel', 'Green Channel', 'Blue Channel')

    all_hist_values = []

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        all_hist_values.extend(hist.flatten())  # Collect for percentile
        ax_hist.plot(hist, color=color, label=channel_labels[i])

    # Compute 95th percentile to avoid extreme spikes
    max_y = np.percentile(all_hist_values, 95)
    ax_hist.set_ylim([0, max_y * 1.1])  # Add a bit of headroom

    ax_hist.set_title('Pixel Value Distribution', fontsize=14)
    ax_hist.set_xlabel('Pixel Intensity', fontsize=12)
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.set_xlim([0, 256])
    ax_hist.legend(loc='upper right')
    ax_hist.grid(True, linestyle='--', alpha=0.6)


    buf_hist = io.BytesIO()
    plt.savefig(buf_hist, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig_hist)
    buf_hist.seek(0)
    hist_base64 = base64.b64encode(buf_hist.getvalue()).decode('utf-8')

    # --- Optional: Log pixel stats ---
    print(f"[Pixel Stats] Mean: {img.mean():.4f}, Min: {img.min()}, Max: {img.max()}, Std: {img.std():.4f}")

    return mask_base64, hist_base64



@app.route('/', methods=['GET', 'POST'])
def upload_image():
    """Handles image uploads and returns classification and segmentation results."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get classification predictions
            predictions = get_predictions(file_path)

            # Get segmentation mask and histogram
            mask_base64, hist_base64 = segment_image(file_path)

            return render_template('result.html', 
                                   predictions=predictions, 
                                   image_path=url_for('static', filename=f'uploads/{filename}'),
                                   mask_base64=mask_base64,
                                   hist_base64=hist_base64)
    
    return render_template('upload.html')
from flask_cors import CORS

# Enable CORS for mobile app communication
CORS(app)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Run predictions
        print("[INFO] Running predictions...")
        predictions = get_predictions(file_path)

        print("[INFO] Running segmentation...")
        mask_base64, hist_base64 = segment_image(file_path)

        response_data = {
            'predictions': predictions,
            'segmentation_mask': mask_base64,
            'pixel_histogram': hist_base64
        }

        print("[INFO] Sending response")
        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] Exception in /predict_api: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


# Run Flask App
# Run Flask App with Waitress
if __name__ == '__main__':
    from waitress import serve
    import socket

    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"Server running at: http://{local_ip}:5000")

    serve(app, host='0.0.0.0', port=5000)


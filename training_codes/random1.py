import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# ==== GPU Setup (for TensorFlow feature extraction only) ====
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]
        )
        print("✅ GPU configured with 3.5 GB memory limit.")
    except RuntimeError as e:
        print("❌ GPU setup error:", e)
else:
    print("⚠️ No GPU detected. Using CPU.")

# ==== Paths ====
data_dir = 'dataset/train'
output_dir = 'rf_results'
os.makedirs(output_dir, exist_ok=True)

# ==== Data Preparation with Augmentation ====
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='sparse', subset='training', shuffle=False
)
val_gen = datagen.flow_from_directory(
    data_dir, target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='sparse', subset='validation', shuffle=False
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)

# ==== Feature Extraction with MobileNetV2 ====
print("🔍 Loading MobileNetV2 for feature extraction...")
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(generator, name):
    features = []
    labels = []
    total_batches = len(generator)
    print(f"\n🔄 Extracting {name} features ({total_batches} batches)...")
    for i, (x_batch, y_batch) in enumerate(generator):
        batch_features = model.predict(x_batch, verbose=0)
        features.append(batch_features)
        labels.append(y_batch)

        percent_done = ((i + 1) / total_batches) * 100
        print(f"\r{name} Progress: {percent_done:.2f}% ({i+1}/{total_batches} batches)", end='')

        if i + 1 == total_batches:
            break
    print(f"\n✅ {name} feature extraction complete.")
    return np.vstack(features), np.concatenate(labels)

start_time = time.time()
X_train, y_train = extract_features(train_gen, "Training")
X_val, y_val = extract_features(val_gen, "Validation")
end_time = time.time()
print(f"⏱️ Feature extraction completed in {(end_time - start_time)/60:.2f} minutes.")

# ==== Random Forest Training ====
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("✅ Random Forest training complete.")

# ==== Save Model ====
model_path = os.path.join(output_dir, 'rf_model.joblib')
joblib.dump(rf_model, model_path)
print(f"💾 Random Forest model saved to '{model_path}'.")

# ==== Evaluation ====
print("\n📊 Evaluating the model...")
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"🎯 Validation Accuracy: {accuracy * 100:.2f}%")

# ==== Confusion Matrix ====
conf_mat = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(18, 16))  # Increased figure size for better cell spacing
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=True, annot_kws={"size": 10})

plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels vertically
plt.yticks(fontsize=10)

plt.tight_layout()  # Prevent label cutoff
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()


# ==== Classification Report ====
report = classification_report(y_val, y_pred, target_names=class_names)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)
print("📄 Classification report saved.")

# ==== Done ====
print("✅ All evaluation artifacts saved in:", output_dir)

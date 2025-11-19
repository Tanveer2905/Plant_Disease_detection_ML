import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf

# ==== Setup ====
tf.get_logger().setLevel('ERROR')
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]
        )
        print("✅ GPU configured.")
    except RuntimeError as e:
        print("❌ GPU setup error:", e)
else:
    print("⚠️ No GPU detected. Using CPU.")

# ==== Parameters and Paths ====
img_size = 224
batch_size = 32
result_dir = 'svm_results'
os.makedirs(result_dir, exist_ok=True)

# ==== Data Preparation ====
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'dataset/train', target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='sparse', subset='training', shuffle=False
)
val_gen = datagen.flow_from_directory(
    'dataset/train', target_size=(img_size, img_size), batch_size=batch_size,
    class_mode='sparse', subset='validation', shuffle=False
)

class_names = list(train_gen.class_indices.keys())
num_classes = len(class_names)

# ==== MobileNetV2 Feature Extractor ====
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# ==== Extract Features ====
def extract_features(generator):
    features = []
    labels = []
    for _ in range(len(generator)):
        imgs, lbls = generator.next()
        feats = feature_extractor.predict(imgs, verbose=0)
        features.append(feats)
        labels.append(lbls)
    return np.vstack(features), np.concatenate(labels)

print("🔍 Extracting features...")
start_time = time.time()
X_train, y_train = extract_features(train_gen)
X_val, y_val = extract_features(val_gen)
print(f"🕒 Feature extraction took {(time.time() - start_time)/60:.2f} minutes.")

# ==== SVM Training ====
print("🧠 Training SVM classifier...")
svm_clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_clf.fit(X_train, y_train)

# ==== Evaluation ====
y_pred = svm_clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ Validation Accuracy: {acc * 100:.2f}%")

# ==== Confusion Matrix ====
conf_mat = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
plt.close()

# ==== Classification Report ====
report = classification_report(y_val, y_pred, target_names=class_names)
print("\nClassification Report:\n", report)

with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ==== Save Accuracy Plot ====
plt.figure()
plt.bar(['SVM Accuracy'], [acc * 100], color='green')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('SVM Accuracy on Validation Set')
plt.savefig(os.path.join(result_dir, 'svm_accuracy.png'))
plt.close()

print(f"📁 All results saved in '{result_dir}' directory.")

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# ==== GPU Setup with 3.5 GB Memory Limit ====
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

# ==== Data Augmentation ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ==== Load Dataset ====
img_size = 224
train_set = train_datagen.flow_from_directory(
    'dataset/train', target_size=(img_size, img_size), batch_size=32,
    class_mode='categorical', subset='training', shuffle=True
)
val_set = val_datagen.flow_from_directory(
    'dataset/train', target_size=(img_size, img_size), batch_size=32,
    class_mode='categorical', subset='validation', shuffle=False
)

num_classes = len(train_set.class_indices)
class_names = list(train_set.class_indices.keys())
print(f"Detected {num_classes} classes: {class_names}")

# ==== Model Definition ====
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

cnn_model = Sequential([
    base_model,
    Flatten(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# ==== Callbacks ====
os.makedirs('models', exist_ok=True)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
model_ckpt = ModelCheckpoint('models/best_mobilenetv2.h5', save_best_only=True, monitor='val_accuracy', verbose=1)

# ==== Training ====
print("\n🔧 Starting enhanced training...")
start_time = time.time()
history = cnn_model.fit(
    train_set,
    validation_data=val_set,
    epochs=30,
    callbacks=[early_stop, reduce_lr, model_ckpt]
)
end_time = time.time()
print(f"🕒 Training completed in {(end_time - start_time)/60:.2f} minutes.")

# ==== Save Final Model ====
cnn_model.save('models/mobilenetv2_finetuned')
print("✅ Final model saved at 'models/mobilenetv2_finetuned'")

# ==== Directory to Save Reports and Plots ====
output_dir = 'model_outputs'
os.makedirs(output_dir, exist_ok=True)

# ==== Accuracy & Loss Plot ====
plt.figure(figsize=(12, 5))
epochs_range = range(len(history.history['loss']))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_plots.png'))
plt.close()
print("📊 Training accuracy and loss graphs saved.")

# ==== Evaluation ====
true_labels = val_set.classes
predictions = np.argmax(cnn_model.predict(val_set), axis=1)

# ==== Confusion Matrix ====
conf_mat = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(18, 16))  # Increased figure size for better cell spacing
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=True, annot_kws={"size": 10})

plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()
print("📌 Confusion matrix saved as PNG.")

# ==== Classification Report ====
report = classification_report(true_labels, predictions, target_names=class_names)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)
print("📄 Classification report saved.")

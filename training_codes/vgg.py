import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import time

# ==== GPU Setup with 3.5 GB Memory Limit ====
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]  # 3.5 GB
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ GPU configured with 3.5 GB memory limit.")
    except RuntimeError as e:
        print("❌ GPU setup error:", e)
else:
    print("⚠️ No GPU detected. Using CPU.")

# ==== Data Augmentation ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_set = train_datagen.flow_from_directory(
    'dataset/train', target_size=(128, 128), batch_size=32,
    class_mode='sparse', subset='training', shuffle=True
)
val_set = val_datagen.flow_from_directory(
    'dataset/train', target_size=(128, 128), batch_size=32,
    class_mode='sparse', subset='validation', shuffle=False
)

num_classes = len(train_set.class_indices)
print(f"Detected {num_classes} classes: {list(train_set.class_indices.keys())}")

# ==== Load VGG16 Base ====
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False

# ==== Add Custom Layers ====
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

vgg_model = Model(inputs=base_model.input, outputs=output)
vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==== Initial Training (Feature Extraction Phase) ====
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

print("\n🔧 Starting initial training...")
start_time = time.time()
history = vgg_model.fit(
    train_set,
    validation_data=val_set,
    epochs=25,
    callbacks=[early_stop, reduce_lr]
)
end_time = time.time()
print(f"🕒 Initial training done in {(end_time - start_time)/60:.2f} minutes.")

# ==== Fine-Tune Top Layers of VGG16 ====
# Unfreeze last 4 conv layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile with lower LR for fine-tuning
vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n🔧 Starting fine-tuning (last 4 conv layers)...")
start_ft = time.time()
history_ft = vgg_model.fit(
    train_set,
    validation_data=val_set,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)
end_ft = time.time()
print(f"🕒 Fine-tuning done in {(end_ft - start_ft)/60:.2f} minutes.")

# ==== Save Fine-Tuned Model ====
os.makedirs('models', exist_ok=True)
vgg_model.save('models/vgg16_finetuned')
print("✅ Fine-tuned model saved at 'models/vgg16_finetuned'")

# ==== Plot Accuracy & Loss ====
plt.figure(figsize=(12, 5))
# Combine both histories
total_epochs = len(history.history['loss']) + len(history_ft.history['loss'])
epochs_range = range(total_epochs)

train_loss = history.history['loss'] + history_ft.history['loss']
val_loss = history.history['val_loss'] + history_ft.history['val_loss']
train_acc = history.history['accuracy'] + history_ft.history['accuracy']
val_acc = history.history['val_accuracy'] + history_ft.history['val_accuracy']

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# ==== Confusion Matrix ====
true_labels = val_set.classes
predictions = np.argmax(vgg_model.predict(val_set), axis=1)
conf_mat = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_set.class_indices.keys(),
            yticklabels=train_set.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ==== Classification Report ====
print("\nClassification Report:\n")
print(classification_report(true_labels, predictions, target_names=train_set.class_indices.keys()))

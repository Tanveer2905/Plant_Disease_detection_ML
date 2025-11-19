import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define dataset paths
IMAGE_DIR = "dataset/segment/images"
MASK_DIR = "dataset/segment/masks"
IMG_HEIGHT, IMG_WIDTH = 256, 256

# Load images and masks
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    
    for img_name, mask_name in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0  # Normalize
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = mask / 255.0  # Normalize
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load dataset
images, masks = load_data(IMAGE_DIR, MASK_DIR)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Define U-Net Model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train model
model = unet_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=20)

# Save the model
model.save("leaf_disease_segmentation_model.h5")

# Evaluate and visualize results
def visualize_predictions(model, X_val, y_val, num_samples=3):
    preds = model.predict(X_val[:num_samples])
    preds = (preds > 0.5).astype(np.uint8)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    for i in range(num_samples):
        axes[i, 0].imshow(X_val[i])
        axes[i, 0].set_title("Original Image")
        axes[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
    plt.show()

visualize_predictions(model, X_val, y_val)
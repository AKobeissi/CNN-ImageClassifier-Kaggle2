import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                     BatchNormalization, Activation, Add, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

with open('/kaggle/input/ift3395-ift6390-identification-maladies-retine/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

images = np.array(train_data['images'], dtype='float32')  # Shape: (num_samples, 28, 28)
labels = np.array(train_data['labels'], dtype='int32')    # Shape: (num_samples,)

print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')

# 1. Preprocess the Images
# Normalize the images to [0,1]
#images /= 255.0 already done in data

# Reshape images to add channel dimension for CNNs
images = images.reshape(-1, 28, 28, 1)  # Shape: (num_samples, 28, 28, 1)

# 2. Visualize Sample Images
fig, axes = plt.subplots(1, 4, figsize=(12,3))
class_names = ['Choroidal Neovascularization', 'Diabetic Macular Edema', 'Drusen', 'Healthy Retina']

for i in range(4):
    idx = np.where(labels == i)[0][0]
    axes[i].imshow(images[idx].reshape(28,28), cmap='gray')
    axes[i].set_title(class_names[i])
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# 3. Label Distribution
plt.figure(figsize=(8,6))
sns.countplot(x=labels)
plt.xticks(ticks=[0,1,2,3], labels=class_names)
plt.xlabel('Retinal Disease')
plt.ylabel('Count')
plt.title('Label Distribution in Training Data')
plt.show()

# Check for class imbalance
unique, counts = np.unique(labels, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Class distribution:", class_counts)

# 4. Shuffle and Split the Data
images, labels = shuffle(images, labels, random_state=42)

# Split into training and validation sets with stratification
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels)

print(f'Number of training examples: {train_labels.shape[0]}')
print(f'Number of validation examples: {val_labels.shape[0]}')

# 5. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Fit the data generator to training data
datagen.fit(train_images)

# 6. Compute Class Weights to Handle Class Imbalance
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

def create_improved_cnn_model(input_shape=(28,28,1), num_classes=4):
    inputs = Input(shape=input_shape)
    
    # First Convolutional Block
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)  # Downsample to (14,14,32)
    x = Dropout(0.25)(x)
    
    # Second Convolutional Block with Residual Connection
    # Main Path
    y = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    y = BatchNormalization()(y)
    y = Conv2D(64, (3,3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D((2,2))(y)  # Downsample to (7,7,64)
    
    # Shortcut Path
    shortcut = Conv2D(64, (1,1), strides=(2,2), padding='same')(x)  # Match dimensions
    shortcut = BatchNormalization()(shortcut)
    
    # Add Shortcut to Main Path
    y = Add()([y, shortcut])  # Both tensors are (7,7,64)
    y = Dropout(0.25)(y)
    
    # Third Convolutional Block with Residual Connection
    # Main Path
    y = Conv2D(128, (3,3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = Conv2D(128, (3,3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    
    # Move MaxPooling after the residual connection
    # Shortcut Path with matching strides
    shortcut = Conv2D(128, (1,1), padding='same')(y)  # No stride here
    shortcut = BatchNormalization()(shortcut)
    
    # Add Shortcut to Main Path
    y = Add()([y, shortcut])  # Both tensors are now (7,7,128)
    
    # Apply pooling after the residual connection
    y = MaxPooling2D((2,2))(y)  # Downsample to (3,3,128)
    y = Dropout(0.25)(y)
    
    # Global Average Pooling
    y = GlobalAveragePooling2D()(y)
    
    # Fully Connected Layers
    y = Dense(256, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    
    y = Dense(128, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)
    
    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(y)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Instantiate the improved model
model_improved = create_improved_cnn_model()

# 7. Compile Model
optimizer = Adam(learning_rate=0.001)
model_improved.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 8. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# 9. Train Model with Data Augmentation
batch_size = 64
epochs = 100

history = model_improved.fit(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=train_labels.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(val_images, val_labels),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# 10. Model Summary
model_improved.summary()

# 11. Plot Training & Validation Accuracy and Loss
def plot_accuracy_loss_chart(history):
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    
    # Accuracy Plot
    ax[0].plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
    ax[0].plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    # Loss Plot
    ax[1].plot(epochs_range, history.history['loss'], label='Training Loss')
    ax[1].plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_accuracy_loss_chart(history)

# 12. Evaluate the Model on Validation Set
val_loss, val_accuracy = model_improved.evaluate(val_images, val_labels, verbose=0)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy*100:.2f}%')

# 13. Confusion Matrix and Classification Report
val_predictions = model_improved.predict(val_images)
val_pred_labels = np.argmax(val_predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(val_labels, val_pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Validation Set')
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(val_labels, val_pred_labels, target_names=class_names))

# 14. Load Test Data from 'test_data.pkl'
with open('/kaggle/input/ift3395-ift6390-identification-maladies-retine/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_images = np.array(test_data['images'], dtype='float32')  # Shape: (num_test_samples, 28, 28)
#test_images /= 255.0  # Normalize
test_images = test_images.reshape(-1, 28, 28, 1)

print(f'Test Images shape: {test_images.shape}')

# 15. Predict on Test Data
test_predictions = model_improved.predict(test_images)
test_pred_labels = np.argmax(test_predictions, axis=1)

# 16. Prepare Submission File
submission = pd.DataFrame({
    'ID': np.arange(1, len(test_pred_labels) + 1),
    'Class': test_pred_labels
})

# Save to CSV
submission.to_csv('submission_CNN_augmented.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

# Display a few predictions
fig, axes = plt.subplots(2,5, figsize=(15,6))
for i, ax in enumerate(axes.flatten()[:10]):
    ax.imshow(test_images[i].reshape(28,28), cmap='gray')
    ax.set_title(f'Predicted: {class_names[test_pred_labels[i]]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

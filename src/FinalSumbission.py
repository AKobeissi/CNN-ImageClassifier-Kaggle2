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

# Preprocess the Images

# Reshape images to add channel dimension for CNNs
images = images.reshape(-1, 28, 28, 1)  # Shape: (num_samples, 28, 28, 1)

fig, axes = plt.subplots(1, 4, figsize=(12,3))
class_names = ['Choroidal Neovascularization', 'Diabetic Macular Edema', 'Drusen', 'Healthy Retina']

for i in range(4):
    idx = np.where(labels == i)[0][0]
    axes[i].imshow(images[idx].reshape(28,28), cmap='gray')
    axes[i].set_title(class_names[i])
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# Data Visualization - Label Distribution
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

# Shuffle and Split the Data
images, labels = shuffle(images, labels, random_state=42)

# Split into training + validation and mini test sets (90% train+val, 10% mini test)
train_val_images, mini_test_images, train_val_labels, mini_test_labels = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels)

# Further split the training + validation set into training and validation sets (80% train, 10% val)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.1111, random_state=42, stratify=train_val_labels)
# (10% of 90% = ~10% validation)

print(f'Number of training examples: {train_labels.shape[0]}')
print(f'Number of validation examples: {val_labels.shape[0]}')
print(f'Number of mini test examples: {mini_test_labels.shape[0]}')

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)


# Fit the data generator to training data
datagen.fit(train_images)

# Compute Class Weights to Handle Class Imbalance
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

def create_cnn_model(input_shape=(28,28,1), num_classes=4):
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

# Instantiate the model
model_improved = create_cnn_model()

# Compile the Model
optimizer = Adam(learning_rate=0.001)
model_improved.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Train the Model with Data Augmentation
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

model_improved.summary()

# Plot Training & Validation Accuracy and Loss
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

# For hyperparameter tuning
def train_resnet_with_params(learning_rate, batch_size, conv_dropout, dense_dropout):
    """
    Train ResNet with specific hyperparameters and return validation accuracy
    """
    model = create_resnet_model(conv_dropout, dense_dropout)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                              restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                 patience=5, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=train_labels.shape[0] // batch_size,
        epochs=30,
        validation_data=(val_images, val_labels),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    
    return max(history.history['val_accuracy'])
  
# # Hyperparameter configurations to test
# configs = [
#     {'lr': 0.001, 'batch_size': 64, 'conv_dropout': 0.25, 'dense_dropout': 0.5},  # baseline
#     {'lr': 0.01, 'batch_size': 64, 'conv_dropout': 0.25, 'dense_dropout': 0.5},
#     {'lr': 0.0001, 'batch_size': 64, 'conv_dropout': 0.25, 'dense_dropout': 0.5},
#     {'lr': 0.001, 'batch_size': 32, 'conv_dropout': 0.25, 'dense_dropout': 0.5},
#     {'lr': 0.001, 'batch_size': 128, 'conv_dropout': 0.25, 'dense_dropout': 0.5},
#     {'lr': 0.001, 'batch_size': 64, 'conv_dropout': 0.1, 'dense_dropout': 0.3},
#     {'lr': 0.001, 'batch_size': 64, 'conv_dropout': 0.4, 'dense_dropout': 0.6}
# ]

# results = []
# for config in configs:
#     val_acc = train_resnet_with_params(
#         config['lr'], 
#         config['batch_size'], 
#         config['conv_dropout'], 
#         config['dense_dropout']
#     )
#     results.append({**config, 'val_accuracy': val_acc})

# results_df = pd.DataFrame(results)
# print("ResNet Hyperparameter Tuning Results:")
# print(results_df.sort_values('val_accuracy', ascending=False))

# Evaluate the Model on Validation Set
val_loss, val_accuracy = model_improved.evaluate(val_images, val_labels, verbose=0)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy*100:.2f}%')

# Confusion Matrix and Classification Report
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

# Evaluate the Model on the Mini Test Set
mini_test_loss, mini_test_accuracy = model_improved.evaluate(mini_test_images, mini_test_labels, verbose=0)
print(f'Mini Test Loss: {mini_test_loss:.4f}')
print(f'Mini Test Accuracy: {mini_test_accuracy*100:.2f}%')

# Predict on the Mini Test Set
mini_test_predictions = model_improved.predict(mini_test_images)
mini_test_pred_labels = np.argmax(mini_test_predictions, axis=1)

# Confusion Matrix for Mini Test Set
cm_mini_test = confusion_matrix(mini_test_labels, mini_test_pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm_mini_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Mini Test Set')
plt.show()

# Classification Report for Mini Test Set
print("Classification Report on Mini Test Set:\n")
print(classification_report(mini_test_labels, mini_test_pred_labels, target_names=class_names))

with open('/kaggle/input/ift3395-ift6390-identification-maladies-retine/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_images = np.array(test_data['images'], dtype='float32')  # Shape: (num_test_samples, 28, 28)
test_images = test_images.reshape(-1, 28, 28, 1)

print(f'Test Images shape: {test_images.shape}')

# Predict on Test Data
test_predictions = model_improved.predict(test_images)
test_pred_labels = np.argmax(test_predictions, axis=1)

submission = pd.DataFrame({
    'ID': np.arange(1, len(test_pred_labels) + 1),
    'Class': test_pred_labels
})

submission.to_csv('submission_CNN_augmented.csv', index=False)

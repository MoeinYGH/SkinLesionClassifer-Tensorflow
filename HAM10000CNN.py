import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
from sklearn.model_selection import train_test_split

# Load and Prepare Dataset
import kagglehub

path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

# Read metadata
metadata_df = pd.read_csv(os.path.join(path, 'HAM10000_metadata.csv'))

# Data Preprocessing
# Map image paths
part1_images = os.listdir(os.path.join(path, 'HAM10000_images_part_1'))
part2_images = os.listdir(os.path.join(path, 'HAM10000_images_part_2'))

image_part_map = {}
for image_id in metadata_df['image_id']:
    if image_id + '.jpg' in part1_images:
        image_part_map[image_id] = 'part_1'
    elif image_id + '.jpg' in part2_images:
        image_part_map[image_id] = 'part_2'
    else:
        image_part_map[image_id] = 'not_found'

metadata_df['image_part'] = metadata_df['image_id'].map(image_part_map)


def get_image_path(row):
    image_file = row['image_id'] + '.jpg'
    if row['image_part'] == 'part_1':
        return os.path.join(path, 'HAM10000_images_part_1', image_file)
    elif row['image_part'] == 'part_2':
        return os.path.join(path, 'HAM10000_images_part_2', image_file)
    else:
        return None


metadata_df['image_path'] = metadata_df.apply(get_image_path, axis=1)

# One-hot encoding for labels
dx_one_hot = pd.get_dummies(metadata_df['dx'], prefix='dx')
metadata_df = pd.concat([metadata_df, dx_one_hot], axis=1)

# Drop unnecessary columns
metadata_df.drop(columns=['image_id', 'image_part', 'sex', 'age', 'dx', 'dx_type', 'localization'], inplace=True)

# Split data by lesion_id to avoid data leakage
unique_lesions = metadata_df['lesion_id'].unique()
train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.3, random_state=42)
valid_lesions, test_lesions = train_test_split(temp_lesions, test_size=0.5, random_state=42)

train_df = metadata_df[metadata_df['lesion_id'].isin(train_lesions)]
valid_df = metadata_df[metadata_df['lesion_id'].isin(valid_lesions)]
test_df = metadata_df[metadata_df['lesion_id'].isin(test_lesions)]

# Drop lesion_id from datasets
train_df = train_df.drop(columns=['lesion_id'])
valid_df = valid_df.drop(columns=['lesion_id'])
test_df = test_df.drop(columns=['lesion_id'])

# Image preprocessing parameters
IMG_HEIGHT = 450 // 4  # Reduced size for faster training
IMG_WIDTH = 600 // 4
BATCH_SIZE = 8


def parse_image_and_label(image_path, label):
    """Loads and preprocesses an image from a file path."""
    raw_image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(raw_image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize pixel values to the [-1, 1] range
    image = (image - 127.5) / 127.5
    return image, label


def create_dataset(df, shuffle=False):
    # Identify the label columns
    label_cols = [col for col in df.columns if col.startswith('dx_')]

    # Extract file paths and labels
    image_paths = df['image_path'].values
    labels = df[label_cols].values.astype(np.float32)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# Create datasets
train_dataset = create_dataset(train_df, shuffle=True)
valid_dataset = create_dataset(valid_df)
test_dataset = create_dataset(test_df)

# CNN Model Definition
model_conv = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),
    MaxPool2D(pool_size=(2, 2)),

    GlobalAveragePooling2D(),

    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.3),

    Dense(7, activation='softmax')
])

model_conv.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_conv = model_conv.fit(train_dataset, epochs=20, validation_data=valid_dataset)

# Save the model
model_conv.save('HAMCancer.keras')

# Evaluate the model
test_loss_conv, test_acc_conv = model_conv.evaluate(test_dataset)
print(f"Test accuracy (CNN): {test_acc_conv}")

# Plotting Results
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axes[0].plot(history_conv.history['loss'], label='Training Loss')
axes[0].plot(history_conv.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy plot
axes[1].plot(history_conv.history['accuracy'], label='Training Accuracy')
axes[1].plot(history_conv.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

# Visualize a sample prediction
for image, label in test_dataset.take(1):
    sample_image = image[0]
    sample_label = label[0]

sample_image_expanded = tf.expand_dims(sample_image, axis=0)
prediction = model_conv.predict(sample_image_expanded)

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow((sample_image * 127.5 + 127.5).numpy().astype(np.uint8))
axes[0].set_title("Test Image")
axes[0].axis("off")

# Prediction probabilities
axes[1].bar(class_labels, prediction[0])
axes[1].set_title("Prediction Probabilities")
axes[1].set_ylabel("Probability")
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim((0, 1))

plt.tight_layout()
plt.show()

# Model summary
model_conv.summary()
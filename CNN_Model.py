import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def vgg16_custom(train_dir, test_dir, image_size=(224, 224), batch_size=32, epochs=10):
    # Verify TensorFlow GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("Using GPU:", physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found. Using CPU.")

    # Load file paths for train and test data
    train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]

    # Debugging: Check file counts
    print(f"Found {len(train_images)} training images and {len(test_images)} testing images.")

    # Generate dummy labels for demonstration (replace with actual labels)
    train_labels = np.zeros(len(train_images))  # Replace with actual class indices
    test_labels = np.zeros(len(test_images))   # Replace with actual class indices

    # Preprocessing function to dynamically load images
    def preprocess_images(image_paths, labels):
        for path, label in zip(image_paths, labels):
            img = tf.keras.preprocessing.image.load_img(path, target_size=image_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
            yield img_array, label

    # Create TensorFlow datasets for dynamic loading
    train_dataset = tf.data.Dataset.from_generator(
        lambda: preprocess_images(train_images, train_labels),
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).batch(batch_size)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: preprocess_images(test_images, test_labels),
        output_signature=(
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).batch(batch_size)

    # Define the model
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=image_size + (3,)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')  # Example: 10 classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with verbose=1 for standard logs (loss, accuracy, etc.)
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        verbose=1  # Enable verbose training logs
    )

    return model

# Example usage:
train_dir = 'Tornet_Dataset_Images/Train'  # Replace with the correct directory
test_dir = 'Tornet_Dataset_Images/Test'    # Replace with the correct directory

trained_model = vgg16_custom(train_dir, test_dir, image_size=(224, 224), batch_size=4, epochs=2)

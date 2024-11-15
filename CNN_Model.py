import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import datetime

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
    ).batch(batch_size).repeat()

    test_dataset = tf.data.Dataset.from_generator(
        lambda: preprocess_images(test_images, test_labels),
        output_signature=( 
            tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).batch(batch_size).repeat()

    # Calculate steps per epoch
    steps_per_epoch = len(train_images) // batch_size
    validation_steps = len(test_images) // batch_size

    # Define the model
    model = models.Sequential([ 
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=image_size + (3,)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # Replace 10 with the number of your classes
    ])

    # Set an initial low learning rate for Adam optimizer
    initial_learning_rate = 0.001  # You can adjust this value

    # Create a learning rate scheduler to decrease the learning rate over time
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: initial_learning_rate * (0.9 ** epoch),  # Decreases LR by 10% every epoch
        verbose=1
    )

    # Compile the model with the Adam optimizer and the adjusted learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TensorBoard log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10,20')

    # Model Summary
    model.summary()

    # Train the model with TensorBoard and LearningRateScheduler callback
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback, lr_scheduler],
        verbose=1  # Simplified output to avoid odd formatting
    )

    return model

# Example usage:
train_dir = 'Tornet_Dataset_Images/Train'  # Replace with the correct directory
test_dir = 'Tornet_Dataset_Images/Test'    # Replace with the correct directory

trained_model = vgg16_custom(train_dir, test_dir, image_size=(112, 112), batch_size=16, epochs=4)

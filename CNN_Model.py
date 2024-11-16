import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def vgg16_custom(train_dir, test_dir, image_size=(224, 224), batch_size=32, epochs=10,
                 initial_learning_rate=0.001, dropout_rate=0.5, optimizer_type='adam', lr_decay_factor=0.9):
    # Verify TensorFlow GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("Using GPU:", physical_devices)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found. Using CPU.")

    # Load training and validation datasets from the directory
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    # Get the number of classes from the training dataset
    num_classes = len(train_dataset.class_names)
    print(f"Detected {num_classes} classes: {train_dataset.class_names}")

    # Preprocessing: Normalizing the pixel values
    normalization_layer = layers.Rescaling(1./255)

    # Prepare datasets for better performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Define the model (VGG16-like architecture)
    model = models.Sequential([
        normalization_layer,
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=image_size + (3,)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Build the model to initialize its input shape
    model.build(input_shape=(None, image_size[0], image_size[1], 3))

    # Choose optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: initial_learning_rate * (lr_decay_factor ** epoch),
        verbose=1
    )

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TensorBoard log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[tensorboard_callback, lr_scheduler],
        verbose=1
    )

    return model

# Define the directories
train_dir = 'Tornet_Dataset_Images/Train'
test_dir = 'Tornet_Dataset_Images/Test'

# Train the model
trained_model = vgg16_custom(train_dir, test_dir, image_size=(112, 112), batch_size=16, epochs=4)

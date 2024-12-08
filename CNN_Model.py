import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os
import datetime
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Resolve encoding errors

def vgg16_custom(train_dir, test_dir, image_size=(224, 224), batch_size=32, epochs=20,
                 initial_learning_rate=0.01, dropout_rate=0.65, optimizer_type='adam', lr_decay_factor=0.95,
                 class_weights=None):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=image_size + (3,))
    base_model.trainable = False

    # Check for GPU availability
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
    class_names = train_dataset.class_names
    print(f"Detected {num_classes} classes: {class_names}")

    # Compute class weights if not provided
    if class_weights is None:
        all_labels = []
        for _, labels in train_dataset:
            all_labels.extend(labels.numpy())
        class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=all_labels)
        class_weights = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Computed class weights: {class_weights}")

    normalization_layer = layers.Rescaling(1./255)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    # Prepare datasets for better performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

    # Build the model
    model = models.Sequential([
        data_augmentation,
        normalization_layer,
        base_model,
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.build(input_shape=(None, image_size[0], image_size[1], 3))

    # Choose optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate) if optimizer_type.lower() == 'adam' else \
                tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: initial_learning_rate * (lr_decay_factor ** epoch), verbose=1)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TensorBoard logging
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.summary()

    # Train the model with class weights
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[tensorboard_callback, lr_scheduler], verbose=1, class_weight=class_weights)

    # Evaluate and display predictions
    evaluate_model(model, test_dataset, class_names)
    return model


def evaluate_model(model, test_dataset, class_names):
    print("\nEvaluating model on the test dataset...\n")
    
    all_labels = []
    all_predictions = []

    # Get predictions and true labels
    for images, labels in test_dataset:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted_labels)
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))


# Example usage
train_dir = 'Tornet_Dataset_Images/Train'
test_dir = 'Tornet_Dataset_Images/Test'


# COMMENT THE FOLLOWING until # ---***--- if you are only using a CPU in your model, otherwise you will bluescreen if you run the model for long. (takes up all threads / cores otherwise...)

# Limit CPU usage to prevent bluescreening
p = psutil.Process()
p.cpu_affinity([1, 2, 3, 4, 5, 6, 7, 8,9,10])
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

# ---***---

trained_model = vgg16_custom(train_dir, test_dir, image_size=(112, 112), batch_size=64, epochs=10)

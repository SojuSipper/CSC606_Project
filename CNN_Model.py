import sys
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import datetime
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') #for some reason I have to have this, I'd get a bunch of errors for some reason otherwise

def vgg16_custom(train_dir, test_dir, image_size=(224, 224), batch_size=32, epochs=20,
                 initial_learning_rate=0.01, dropout_rate=0.65, optimizer_type='adam', lr_decay_factor=0.95):
    

    # Checks for GPU, if found use the GPU for the model, other wise use the CPU... (lmao have fun with this if you did not UNCOMMENT THE LAST COUPLE LINES MENTIONED IN THE CODE)
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

    normalization_layer = layers.Rescaling(1./255)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    # Prepare datasets for better performance, found this online to help mitigate data balance (uses autotune from keras)
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Vgg-16 model 
    model = models.Sequential([
        data_augmentation,
        normalization_layer,

        # 3x3 filter
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=image_size + (3,)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

        #3x3 filter
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

        #3x3 filter
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  

        # 3x3 filter
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),

        #3x3 filter
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),

        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.build(input_shape=(None, image_size[0], image_size[1], 3))

    # Choose optimizer either sgd or adam, couldnt be bothered to look into other ones :^) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate) if optimizer_type.lower() == 'adam' else \
                tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: initial_learning_rate * (lr_decay_factor ** epoch), verbose=1)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TensorBoard logging yoinked from one of my other classes 
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.summary()

    # Train the model
    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[tensorboard_callback, lr_scheduler], verbose=1)

    # Evaluate and display predictions
    evaluate_model(model, test_dataset, class_names)
    return model


# had to get help with this online, I could not figure this out for the life of me
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



# Define directories CHANGE ME TO WHERE EVER YOU ARE STORING THE IMAGES SAVED FROM THE SCRIPT
train_dir = 'Tornet_Dataset_Images/Train'
test_dir = 'Tornet_Dataset_Images/Test'

# UNCOMMENT THE FOLLOWING until # ---***--- if you are only using a CPU in your model, otherwise you will bluescreen if you run the model for long. (takes up all threads / cores otherwise...)
# Limit CPU usage to prevent bluescreening
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"


p = psutil.Process()
p.cpu_affinity([1, 2, 3, 4, 5, 6, 7, 8])
p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

# ---***---


# Train and evaluate the model
trained_model = vgg16_custom(train_dir, test_dir, image_size=(224, 224), batch_size=64, epochs=20)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
import datetime
import numpy as np
import os

def CNN_with_tensorboard(epochs=5):
    # Paths to your training and test directories
    train_dir = 'Train'
    test_dir = 'Test'

    # Image parameters
    img_height, img_width = 224, 224  # VGG-16 uses 224x224 input images
    batch_size = 32
    num_classes = len(os.listdir(train_dir))  # Number of classes based on directories in 'Train'

    # preping the data for input 
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # VGG-16 Architecture 
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
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
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



    # TensorBoard log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #should be stored locally 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10,20')
    
    # Train the model
    model.summary()
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=[tensorboard_callback]
    )

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')

    return model

trained_model = CNN_with_tensorboard(epochs=5)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to VGG-16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def make_prediction_on_image(model, img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    predicted_label = np.argmax(prediction, axis=1)[0]
    return predicted_label

img_path = '.jpg' #when an image is made put this as the .jpg 
predicted_label = make_prediction_on_image(trained_model, img_path)
print(f"The predicted label for the image is: {predicted_label}")


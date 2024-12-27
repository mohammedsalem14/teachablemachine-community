import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(
    'data/train',  # Replace with the actual path to your training directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)  # Adjust epochs as needed

# Save the trained model
model.save('dog_cat_model.h5')  # Save the model to an HDF5 file

# Load and preprocess a new image
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path/to/your/image.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

# Make a prediction
prediction = model.predict(x)

# Print the prediction
if prediction[0][0] > 0.5:
    print('Predicted class: Dog')
else:
    print('Predicted class: Cat')
